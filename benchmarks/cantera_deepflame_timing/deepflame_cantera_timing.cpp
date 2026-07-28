#include "cantera/base/Solution.h"
#include "cantera/base/config.h"
#ifdef DFODE_CANTERA_HAS_SOLVER_STATS
#include "cantera/base/AnyMap.h"
#endif
#include "cantera/numerics/Integrator.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/zeroD/Reactor.h"
#include "cantera/zeroD/ReactorNet.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string mechanism;
    std::string phase;
    std::string fuel = "H2:1";
    std::string oxidizer = "O2:1,N2:3.76";
    double temperature = 300.0;
    double pressure = 101325.0;
    double phi = 1.0;
    std::vector<double> dtValues{1e-6};
    size_t states = 11;
    size_t repeats = 5;
    size_t warmup = 1;
    double rtol = 1e-6;
    double atol = 1e-10;
    fs::path outputDir = ".";
    bool writeCellCsv = true;
    bool profileAdvance = false;
};

struct CellState {
    double progress = 0.0;
    double temperature = 0.0;
    double pressure = 0.0;
    std::vector<double> massFractions;
};

struct RhsEvalTiming {
    double solverTime = 0.0;
    double callbackUs = 0.0;
    double reactorUs = 0.0;
};

struct CellTiming {
    double dt = 0.0;
    size_t repeat = 0;
    size_t stateIndex = 0;
    double progress = 0.0;
    double temperature = 0.0;
    double pressure = 0.0;
    double density = 0.0;
    int rhsEvaluations = 0;
    int lastOrder = 0;
    size_t profiledRhsCalls = 0;
    double setStateUs = 0.0;
    double syncStateUs = 0.0;
    double advanceUs = 0.0;
    double readbackUs = 0.0;
    double resetTimeUs = 0.0;
    double sourceUs = 0.0;
    double totalUs = 0.0;
    double rhsCallbackUs = 0.0;
    double reactorEvalUs = 0.0;
    double reactorNetOverheadUs = 0.0;
    double cvodesOverheadUs = 0.0;
    long int cvodesSteps = 0;
    long int cvodesRhsEvals = 0;
    long int cvodesLinearRhsEvals = 0;
    long int cvodesJacobianEvals = 0;
    long int cvodesLinearSetups = 0;
    long int cvodesNonlinearIterations = 0;
    long int cvodesNonlinearConvFails = 0;
    long int cvodesErrorTestFails = 0;
    double maxAbsDeltaY = 0.0;
    double massSumError = 0.0;
    double qdot = 0.0;
};

struct Distribution {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double maximum = 0.0;
};

static double elapsedUs(const Clock::time_point& start, const Clock::time_point& end)
{
    return std::chrono::duration<double, std::micro>(end - start).count();
}

class TimedReactor : public Cantera::Reactor
{
public:
#ifdef DFODE_CANTERA_HAS_SOLVER_STATS
    explicit TimedReactor(const std::shared_ptr<Cantera::Solution>& solution)
        : Cantera::Reactor(solution, false)
    {
    }
#else
    TimedReactor() = default;
#endif

    void setProfiling(bool enabled)
    {
        m_enabled = enabled;
    }

    void resetProfile()
    {
        m_calls = 0;
        m_totalUs = 0.0;
    }

    void eval(double t, double* lhs, double* rhs) override
    {
        if (!m_enabled) {
            Cantera::Reactor::eval(t, lhs, rhs);
            return;
        }
        const auto start = Clock::now();
        Cantera::Reactor::eval(t, lhs, rhs);
        m_totalUs += elapsedUs(start, Clock::now());
        ++m_calls;
    }

    size_t profiledCalls() const
    {
        return m_calls;
    }

    double profiledTotalUs() const
    {
        return m_totalUs;
    }

private:
    bool m_enabled = false;
    size_t m_calls = 0;
    double m_totalUs = 0.0;
};

class TimedReactorNet : public Cantera::ReactorNet
{
public:
    void setProfiling(bool enabled)
    {
        m_enabled = enabled;
    }

    void attachReactor(const TimedReactor* reactor)
    {
        m_reactor = reactor;
    }

    void resetProfile()
    {
        m_totalUs = 0.0;
        m_samples.clear();
        m_samples.reserve(512);
    }

    void eval(double t, double* y, double* ydot, double* p) override
    {
        if (!m_enabled) {
            Cantera::ReactorNet::eval(t, y, ydot, p);
            return;
        }

        const double reactorBefore = m_reactor ? m_reactor->profiledTotalUs() : 0.0;
        const auto start = Clock::now();
        Cantera::ReactorNet::eval(t, y, ydot, p);
        const double callbackUs = elapsedUs(start, Clock::now());
        const double reactorAfter = m_reactor ? m_reactor->profiledTotalUs() : 0.0;
        const double reactorUs = reactorAfter - reactorBefore;

        m_totalUs += callbackUs;
        m_samples.push_back({t, callbackUs, reactorUs});
    }

    size_t profiledCalls() const
    {
        return m_samples.size();
    }

    double profiledTotalUs() const
    {
        return m_totalUs;
    }

    const std::vector<RhsEvalTiming>& samples() const
    {
        return m_samples;
    }

private:
    bool m_enabled = false;
    const TimedReactor* m_reactor = nullptr;
    double m_totalUs = 0.0;
    std::vector<RhsEvalTiming> m_samples;
};

static std::vector<double> parseDoubles(const std::string& text)
{
    std::vector<double> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    if (values.empty()) {
        throw std::runtime_error("Expected at least one comma-separated number");
    }
    return values;
}

static void printHelp(const char* program)
{
    std::cout
        << "Usage: " << program << " --mechanism FILE [options]\n\n"
        << "Options:\n"
        << "  --phase NAME            YAML phase name (default: first phase)\n"
        << "  --fuel COMPOSITION      Fuel composition (default: H2:1)\n"
        << "  --oxidizer COMPOSITION  Oxidizer composition (default: O2:1,N2:3.76)\n"
        << "  --temperature K         Unburned temperature (default: 300)\n"
        << "  --pressure PA           Pressure (default: 101325)\n"
        << "  --phi VALUE             Equivalence ratio (default: 1)\n"
        << "  --dt LIST               Comma-separated integration intervals\n"
        << "  --states N              Flame-progress states (default: 11)\n"
        << "  --repeats N             Measured passes per dt (default: 5)\n"
        << "  --warmup N              Unmeasured passes per dt (default: 1)\n"
        << "  --rtol VALUE            CVODES relative tolerance (default: 1e-6)\n"
        << "  --atol VALUE            CVODES absolute tolerance (default: 1e-10)\n"
        << "  --output-dir DIR        Output directory (default: current directory)\n"
        << "  --no-cell-csv           Do not write per-cell timing rows\n"
        << "  --profile-advance       Profile every RHS callback inside advance()\n"
        << "  --help                  Show this help\n";
}

static Options parseOptions(int argc, char** argv)
{
    Options options;
    auto requireValue = [&](int& index) -> std::string {
        if (index + 1 >= argc) {
            throw std::runtime_error(std::string("Missing value for ") + argv[index]);
        }
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mechanism") {
            options.mechanism = requireValue(i);
        } else if (arg == "--phase") {
            options.phase = requireValue(i);
        } else if (arg == "--fuel") {
            options.fuel = requireValue(i);
        } else if (arg == "--oxidizer") {
            options.oxidizer = requireValue(i);
        } else if (arg == "--temperature") {
            options.temperature = std::stod(requireValue(i));
        } else if (arg == "--pressure") {
            options.pressure = std::stod(requireValue(i));
        } else if (arg == "--phi") {
            options.phi = std::stod(requireValue(i));
        } else if (arg == "--dt") {
            options.dtValues = parseDoubles(requireValue(i));
        } else if (arg == "--states") {
            options.states = std::stoul(requireValue(i));
        } else if (arg == "--repeats") {
            options.repeats = std::stoul(requireValue(i));
        } else if (arg == "--warmup") {
            options.warmup = std::stoul(requireValue(i));
        } else if (arg == "--rtol") {
            options.rtol = std::stod(requireValue(i));
        } else if (arg == "--atol") {
            options.atol = std::stod(requireValue(i));
        } else if (arg == "--output-dir") {
            options.outputDir = requireValue(i);
        } else if (arg == "--no-cell-csv") {
            options.writeCellCsv = false;
        } else if (arg == "--profile-advance") {
            options.profileAdvance = true;
        } else if (arg == "--help") {
            printHelp(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.mechanism.empty()) {
        throw std::runtime_error("--mechanism is required");
    }
    if (options.states < 2 || options.repeats == 0) {
        throw std::runtime_error("--states must be >= 2 and --repeats must be >= 1");
    }
    for (double dt : options.dtValues) {
        if (!(dt > 0.0)) {
            throw std::runtime_error("All dt values must be positive");
        }
    }
    return options;
}

static std::string configureCanteraDataDirectory()
{
    const char* configured = std::getenv("CANTERA_DATA");
    if (configured && configured[0] != '\0') {
        return configured;
    }

    const char* condaPrefix = std::getenv("CONDA_PREFIX");
    if (!condaPrefix || condaPrefix[0] == '\0') {
        return "";
    }

    const fs::path candidate = fs::path(condaPrefix) / "share/cantera/data";
    if (!fs::is_regular_file(candidate / "element-standard-entropies.yaml")) {
        return "";
    }

    const std::string value = candidate.string();
    if (setenv("CANTERA_DATA", value.c_str(), 0) != 0) {
        throw std::runtime_error("Failed to set CANTERA_DATA=" + value);
    }
    return value;
}

static std::shared_ptr<Cantera::Solution> makeSolution(const Options& options)
{
    return Cantera::newSolution(options.mechanism, options.phase, "none");
}

static std::vector<CellState> makeFlameProgressStates(
    const Options& options,
    double& burnedTemperature)
{
    auto solution = makeSolution(options);
    auto gas = solution->thermo();
    gas->setState_TP(options.temperature, options.pressure);
    gas->setEquivalenceRatio(options.phi, options.fuel, options.oxidizer);

    std::vector<double> unburned(gas->nSpecies());
    std::vector<double> burned(gas->nSpecies());
    gas->getMassFractions(unburned.data());
    gas->equilibrate("HP");
    burnedTemperature = gas->temperature();
    gas->getMassFractions(burned.data());

    std::vector<CellState> states;
    states.reserve(options.states);
    for (size_t i = 0; i < options.states; ++i) {
        const double progress = static_cast<double>(i) / (options.states - 1);
        CellState state;
        state.progress = progress;
        state.temperature = (1.0 - progress) * options.temperature
            + progress * burnedTemperature;
        state.pressure = options.pressure;
        state.massFractions.resize(gas->nSpecies());
        for (size_t k = 0; k < gas->nSpecies(); ++k) {
            state.massFractions[k] = (1.0 - progress) * unburned[k]
                + progress * burned[k];
        }
        const double sum = std::accumulate(
            state.massFractions.begin(), state.massFractions.end(), 0.0);
        for (double& value : state.massFractions) {
            value /= sum;
        }
        states.push_back(std::move(state));
    }
    return states;
}

static CellTiming runCell(
    const CellState& state,
    double dt,
    size_t repeat,
    size_t stateIndex,
    const std::vector<double>& formationEnthalpies,
    Cantera::ThermoPhase& gas,
    TimedReactor& reactor,
    TimedReactorNet& network,
    std::vector<double>& outputMassFractions,
    std::vector<double>& reactionRates)
{
    CellTiming timing;
    timing.dt = dt;
    timing.repeat = repeat;
    timing.stateIndex = stateIndex;
    timing.progress = state.progress;
    timing.temperature = state.temperature;
    timing.pressure = state.pressure;

    const auto totalStart = Clock::now();

    auto start = Clock::now();
    gas.setState_TPY(
        state.temperature,
        state.pressure,
        state.massFractions.data());
    timing.density = gas.density();
    auto end = Clock::now();
    timing.setStateUs = elapsedUs(start, end);

    start = Clock::now();
    reactor.syncState();
    end = Clock::now();
    timing.syncStateUs = elapsedUs(start, end);

    reactor.resetProfile();
    network.resetProfile();
    start = Clock::now();
    network.advance(dt);
    end = Clock::now();
    timing.advanceUs = elapsedUs(start, end);
    timing.rhsEvaluations = network.integrator().nEvals();
    timing.lastOrder = network.integrator().lastOrder();
#ifdef DFODE_CANTERA_HAS_SOLVER_STATS
    const auto solverStats = network.solverStats();
    timing.cvodesSteps = solverStats.at("steps").asInt();
    timing.cvodesRhsEvals = solverStats.at("rhs_evals").asInt();
    timing.cvodesLinearRhsEvals = solverStats.at("lin_rhs_evals").asInt();
    timing.cvodesJacobianEvals = solverStats.at("jac_evals").asInt();
    timing.cvodesLinearSetups = solverStats.at("lin_solve_setups").asInt();
    timing.cvodesNonlinearIterations = solverStats.at("nonlinear_iters").asInt();
    timing.cvodesNonlinearConvFails = solverStats.at("nonlinear_conv_fails").asInt();
    timing.cvodesErrorTestFails = solverStats.at("err_test_fails").asInt();
#endif
    timing.profiledRhsCalls = network.profiledCalls();
    timing.rhsCallbackUs = network.profiledTotalUs();
    timing.reactorEvalUs = reactor.profiledTotalUs();
    timing.reactorNetOverheadUs = std::max(
        0.0, timing.rhsCallbackUs - timing.reactorEvalUs);
    timing.cvodesOverheadUs = std::max(
        0.0, timing.advanceUs - timing.rhsCallbackUs);

    start = Clock::now();
    reactor.restoreState();
    gas.getMassFractions(outputMassFractions.data());
    end = Clock::now();
    timing.readbackUs = elapsedUs(start, end);

    start = Clock::now();
    network.setInitialTime(0.0);
    end = Clock::now();
    timing.resetTimeUs = elapsedUs(start, end);

    start = Clock::now();
    double massSum = 0.0;
    for (size_t k = 0; k < outputMassFractions.size(); ++k) {
        reactionRates[k] = timing.density
            * (outputMassFractions[k] - state.massFractions[k]) / dt;
        timing.qdot -= formationEnthalpies[k] * reactionRates[k];
        timing.maxAbsDeltaY = std::max(
            timing.maxAbsDeltaY,
            std::abs(outputMassFractions[k] - state.massFractions[k]));
        massSum += outputMassFractions[k];
    }
    timing.massSumError = std::abs(massSum - 1.0);
    end = Clock::now();
    timing.sourceUs = elapsedUs(start, end);

    timing.totalUs = elapsedUs(totalStart, Clock::now());
    return timing;
}

static double quantile(std::vector<double> values, double q)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double weight = position - lower;
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

static Distribution distribution(const std::vector<double>& values)
{
    Distribution result;
    if (values.empty()) {
        return result;
    }
    result.mean = std::accumulate(values.begin(), values.end(), 0.0)
        / static_cast<double>(values.size());
    result.p50 = quantile(values, 0.50);
    result.p95 = quantile(values, 0.95);
    result.p99 = quantile(values, 0.99);
    result.maximum = *std::max_element(values.begin(), values.end());
    return result;
}

template<class Getter>
static std::vector<double> collect(
    const std::vector<CellTiming>& records,
    Getter getter)
{
    std::vector<double> values;
    values.reserve(records.size());
    for (const auto& record : records) {
        values.push_back(getter(record));
    }
    return values;
}

static void writeCellHeader(std::ofstream& stream)
{
    stream
        << "dt,repeat,state_index,progress,temperature,pressure,density,"
        << "rhs_evaluations,last_order,set_state_us,sync_state_us,advance_us,"
        << "readback_us,reset_time_us,source_us,total_us,max_abs_delta_y,"
        << "mass_sum_error,qdot,profiled_rhs_calls,rhs_callback_us,"
        << "reactor_eval_us,reactornet_overhead_us,cvodes_overhead_us\n";
}

static void writeCell(std::ofstream& stream, const CellTiming& value)
{
    stream << std::setprecision(17)
        << value.dt << ',' << value.repeat << ',' << value.stateIndex << ','
        << value.progress << ',' << value.temperature << ',' << value.pressure << ','
        << value.density << ',' << value.rhsEvaluations << ',' << value.lastOrder << ','
        << value.setStateUs << ',' << value.syncStateUs << ',' << value.advanceUs << ','
        << value.readbackUs << ',' << value.resetTimeUs << ',' << value.sourceUs << ','
        << value.totalUs << ',' << value.maxAbsDeltaY << ',' << value.massSumError << ','
        << value.qdot << ',' << value.profiledRhsCalls << ',' << value.rhsCallbackUs << ','
        << value.reactorEvalUs << ',' << value.reactorNetOverheadUs << ','
        << value.cvodesOverheadUs << '\n';
}

static void writeSolverStatsHeader(std::ofstream& stream)
{
    stream
        << "dt,repeat,state_index,progress,steps,rhs_evals,lin_rhs_evals,"
        << "total_rhs_evals,jac_evals,lin_solve_setups,nonlinear_iters,"
        << "nonlinear_conv_fails,err_test_fails,profiled_rhs_calls,"
        << "profiled_minus_native_rhs\n";
}

static void writeSolverStats(std::ofstream& stream, const CellTiming& value)
{
    const long int totalRhs = value.cvodesRhsEvals + value.cvodesLinearRhsEvals;
    const long int callbackDifference = static_cast<long int>(value.profiledRhsCalls)
        - totalRhs;
    stream << std::setprecision(17)
        << value.dt << ',' << value.repeat << ',' << value.stateIndex << ','
        << value.progress << ',' << value.cvodesSteps << ','
        << value.cvodesRhsEvals << ',' << value.cvodesLinearRhsEvals << ','
        << totalRhs << ',' << value.cvodesJacobianEvals << ','
        << value.cvodesLinearSetups << ',' << value.cvodesNonlinearIterations << ','
        << value.cvodesNonlinearConvFails << ',' << value.cvodesErrorTestFails << ','
        << value.profiledRhsCalls << ',' << callbackDifference << '\n';
}

static void writeRhsHeader(std::ofstream& stream)
{
    stream
        << "dt,repeat,state_index,progress,rhs_index,solver_time,callback_us,"
        << "reactor_eval_us,reactornet_overhead_us\n";
}

static void writeRhsSamples(
    std::ofstream& stream,
    const CellTiming& cell,
    const std::vector<RhsEvalTiming>& samples)
{
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        stream << std::setprecision(17)
            << cell.dt << ',' << cell.repeat << ',' << cell.stateIndex << ','
            << cell.progress << ',' << i << ',' << sample.solverTime << ','
            << sample.callbackUs << ',' << sample.reactorUs << ','
            << std::max(0.0, sample.callbackUs - sample.reactorUs) << '\n';
    }
}

static void writeSummaryHeader(std::ofstream& stream)
{
    stream
        << "dt,calls,mean_total_us,p50_total_us,p95_total_us,p99_total_us,"
        << "max_total_us,mean_set_state_us,mean_sync_state_us,mean_advance_us,"
        << "p50_advance_us,p95_advance_us,p99_advance_us,max_advance_us,"
        << "mean_readback_us,mean_reset_time_us,mean_source_us,advance_fraction,"
        << "mean_rhs_evaluations,p95_rhs_evaluations,max_rhs_evaluations,"
        << "mean_last_order,mean_profiled_rhs_calls,mean_rhs_callback_us,"
        << "rhs_callback_fraction,mean_reactor_eval_us,reactor_eval_fraction,"
        << "mean_reactornet_overhead_us,mean_cvodes_overhead_us,"
        << "cvodes_overhead_fraction,mean_cvodes_steps,mean_cvodes_rhs_evals,"
        << "mean_cvodes_linear_rhs_evals,mean_cvodes_jacobian_evals,"
        << "mean_cvodes_linear_setups,mean_cvodes_nonlinear_iterations,"
        << "mean_cvodes_nonlinear_conv_fails,mean_cvodes_error_test_fails,"
        << "mean_callback_minus_native_rhs,max_abs_delta_y,max_mass_sum_error\n";
}

static void writeSummary(
    std::ofstream& stream,
    double dt,
    const std::vector<CellTiming>& records)
{
    const auto total = distribution(collect(records, [](const auto& x) { return x.totalUs; }));
    const auto setState = distribution(collect(records, [](const auto& x) { return x.setStateUs; }));
    const auto syncState = distribution(collect(records, [](const auto& x) { return x.syncStateUs; }));
    const auto advance = distribution(collect(records, [](const auto& x) { return x.advanceUs; }));
    const auto readback = distribution(collect(records, [](const auto& x) { return x.readbackUs; }));
    const auto reset = distribution(collect(records, [](const auto& x) { return x.resetTimeUs; }));
    const auto source = distribution(collect(records, [](const auto& x) { return x.sourceUs; }));
    const auto rhs = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.rhsEvaluations);
    }));
    const auto order = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.lastOrder);
    }));
    const auto profiledRhs = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.profiledRhsCalls);
    }));
    const auto rhsCallback = distribution(collect(records, [](const auto& x) {
        return x.rhsCallbackUs;
    }));
    const auto reactorEval = distribution(collect(records, [](const auto& x) {
        return x.reactorEvalUs;
    }));
    const auto reactorNetOverhead = distribution(collect(records, [](const auto& x) {
        return x.reactorNetOverheadUs;
    }));
    const auto cvodesOverhead = distribution(collect(records, [](const auto& x) {
        return x.cvodesOverheadUs;
    }));
    const auto cvodesSteps = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesSteps);
    }));
    const auto cvodesRhs = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesRhsEvals);
    }));
    const auto cvodesLinearRhs = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesLinearRhsEvals);
    }));
    const auto cvodesJacobian = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesJacobianEvals);
    }));
    const auto cvodesLinearSetups = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesLinearSetups);
    }));
    const auto cvodesNonlinear = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesNonlinearIterations);
    }));
    const auto cvodesNonlinearFails = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesNonlinearConvFails);
    }));
    const auto cvodesErrorFails = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.cvodesErrorTestFails);
    }));
    const auto callbackDifference = distribution(collect(records, [](const auto& x) {
        return static_cast<double>(x.profiledRhsCalls)
            - static_cast<double>(x.cvodesRhsEvals + x.cvodesLinearRhsEvals);
    }));
    const auto deltaY = distribution(collect(records, [](const auto& x) { return x.maxAbsDeltaY; }));
    const auto massError = distribution(collect(records, [](const auto& x) { return x.massSumError; }));

    stream << std::setprecision(17)
        << dt << ',' << records.size() << ','
        << total.mean << ',' << total.p50 << ',' << total.p95 << ','
        << total.p99 << ',' << total.maximum << ','
        << setState.mean << ',' << syncState.mean << ',' << advance.mean << ','
        << advance.p50 << ',' << advance.p95 << ',' << advance.p99 << ','
        << advance.maximum << ',' << readback.mean << ',' << reset.mean << ','
        << source.mean << ',' << (total.mean > 0.0 ? advance.mean / total.mean : 0.0) << ','
        << rhs.mean << ',' << rhs.p95 << ',' << rhs.maximum << ',' << order.mean << ','
        << profiledRhs.mean << ',' << rhsCallback.mean << ','
        << (advance.mean > 0.0 ? rhsCallback.mean / advance.mean : 0.0) << ','
        << reactorEval.mean << ','
        << (advance.mean > 0.0 ? reactorEval.mean / advance.mean : 0.0) << ','
        << reactorNetOverhead.mean << ',' << cvodesOverhead.mean << ','
        << (advance.mean > 0.0 ? cvodesOverhead.mean / advance.mean : 0.0) << ','
        << cvodesSteps.mean << ',' << cvodesRhs.mean << ',' << cvodesLinearRhs.mean << ','
        << cvodesJacobian.mean << ',' << cvodesLinearSetups.mean << ','
        << cvodesNonlinear.mean << ',' << cvodesNonlinearFails.mean << ','
        << cvodesErrorFails.mean << ',' << callbackDifference.mean << ','
        << deltaY.maximum << ',' << massError.maximum << '\n';

    std::cout << std::scientific << std::setprecision(3)
        << "dt=" << dt
        << " calls=" << records.size()
        << " total_mean=" << total.mean << " us"
        << " total_p95=" << total.p95 << " us"
        << " advance_mean=" << advance.mean << " us"
        << " advance_share=" << 100.0 * advance.mean / total.mean << "%"
        << " rhs_mean=" << rhs.mean
        << " rhs_p95=" << rhs.p95
        << " order_mean=" << order.mean
        << '\n';
}

int main(int argc, char** argv)
{
    try {
        const Options options = parseOptions(argc, argv);
        const std::string canteraData = configureCanteraDataDirectory();
        fs::create_directories(options.outputDir);

        double burnedTemperature = 0.0;
        const auto states = makeFlameProgressStates(options, burnedTemperature);
        auto probe = makeSolution(options);
        const size_t speciesCount = probe->thermo()->nSpecies();

        std::vector<double> formationEnthalpies(speciesCount);
        for (size_t k = 0; k < speciesCount; ++k) {
            formationEnthalpies[k] = probe->thermo()->Hf298SS(k)
                / probe->thermo()->molecularWeight(k);
        }

        std::ofstream metadata(options.outputDir / "metadata.txt");
        metadata << std::setprecision(17)
            << "cantera_version=" << CANTERA_VERSION << '\n'
            << "cantera_data=" << canteraData << '\n'
            << "mechanism=" << options.mechanism << '\n'
            << "phase=" << options.phase << '\n'
            << "species=" << speciesCount << '\n'
            << "fuel=" << options.fuel << '\n'
            << "oxidizer=" << options.oxidizer << '\n'
            << "phi=" << options.phi << '\n'
            << "unburned_temperature=" << options.temperature << '\n'
            << "burned_temperature=" << burnedTemperature << '\n'
            << "pressure=" << options.pressure << '\n'
            << "states=" << options.states << '\n'
            << "repeats=" << options.repeats << '\n'
            << "warmup=" << options.warmup << '\n'
            << "profile_advance=" << options.profileAdvance << '\n'
#ifdef DFODE_CANTERA_HAS_SOLVER_STATS
            << "native_solver_stats=1\n"
#else
            << "native_solver_stats=0\n"
#endif
            << "rtol=" << options.rtol << '\n'
            << "atol=" << options.atol << '\n';

        std::ofstream summary(options.outputDir / "summary.csv");
        writeSummaryHeader(summary);

        std::ofstream cells;
        if (options.writeCellCsv) {
            cells.open(options.outputDir / "cell_timings.csv");
            writeCellHeader(cells);
        }

        std::ofstream rhsEvaluations;
        if (options.profileAdvance) {
            rhsEvaluations.open(options.outputDir / "rhs_evaluations.csv");
            writeRhsHeader(rhsEvaluations);
        }

        std::ofstream solverStatistics;
#ifdef DFODE_CANTERA_HAS_SOLVER_STATS
        solverStatistics.open(options.outputDir / "solver_stats.csv");
        writeSolverStatsHeader(solverStatistics);
#endif

        std::cout
            << "Cantera " << CANTERA_VERSION
            << ", species=" << speciesCount
            << ", flame-progress states=" << states.size()
            << ", burned T=" << burnedTemperature << " K\n";

        for (double dt : options.dtValues) {
            auto working = makeSolution(options);
            auto gas = working->thermo();
            gas->setState_TPY(
                states.front().temperature,
                states.front().pressure,
                states.front().massFractions.data());

#ifdef DFODE_CANTERA_HAS_SOLVER_STATS
            TimedReactor reactor(working);
#else
            TimedReactor reactor;
#endif
            reactor.setProfiling(options.profileAdvance);
            reactor.setEnergy(0);
#ifndef DFODE_CANTERA_HAS_SOLVER_STATS
            reactor.insert(working);
#endif
            TimedReactorNet network;
            network.setProfiling(options.profileAdvance);
            network.attachReactor(&reactor);
            network.addReactor(reactor);
            network.setTolerances(options.rtol, options.atol);

            std::vector<double> outputMassFractions(speciesCount);
            std::vector<double> reactionRates(speciesCount);

            for (size_t warmup = 0; warmup < options.warmup; ++warmup) {
                for (size_t stateIndex = 0; stateIndex < states.size(); ++stateIndex) {
                    (void) runCell(
                        states[stateIndex],
                        dt,
                        warmup,
                        stateIndex,
                        formationEnthalpies,
                        *gas,
                        reactor,
                        network,
                        outputMassFractions,
                        reactionRates);
                }
            }

            std::vector<CellTiming> records;
            records.reserve(options.repeats * states.size());
            for (size_t repeat = 0; repeat < options.repeats; ++repeat) {
                for (size_t stateIndex = 0; stateIndex < states.size(); ++stateIndex) {
                    auto record = runCell(
                        states[stateIndex],
                        dt,
                        repeat,
                        stateIndex,
                        formationEnthalpies,
                        *gas,
                        reactor,
                        network,
                        outputMassFractions,
                        reactionRates);
                    if (options.writeCellCsv) {
                        writeCell(cells, record);
                    }
                    if (options.profileAdvance) {
                        writeRhsSamples(rhsEvaluations, record, network.samples());
                    }
#ifdef DFODE_CANTERA_HAS_SOLVER_STATS
                    writeSolverStats(solverStatistics, record);
#endif
                    records.push_back(record);
                }
            }
            writeSummary(summary, dt, records);
        }

        std::cout << "Wrote timing results to " << options.outputDir << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
