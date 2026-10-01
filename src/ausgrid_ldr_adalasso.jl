# ============================================================
# Cell 1
# ============================================================
# ============================================================
# 0. Pacotes
# ============================================================

import Pkg

required_packages = [
    "CSV",
    "DataFrames",
    "JuMP",
    "Gurobi",
    "Plots",
    "StatsPlots",
]

for pkg in required_packages
    if Base.find_package(pkg) === nothing
        println("Instalando pacote: ", pkg)
        Pkg.add(pkg)
    end
end

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using Printf
using Downloads
using JuMP
using Gurobi
using Plots
using StatsPlots
import MathOptInterface as MOI

Random.seed!(2026)

println("Pacotes carregados.")

# ============================================================
# Cell 2
# ============================================================
# ============================================================
# 1. Parâmetros
# ============================================================

const DT_HOURS = 0.5

# Bateria
const BMAX = 10.0                 # kWh
const B0 = 5.0                    # kWh
const PMAX_KW = 5.0               # kW
const EMAX = PMAX_KW * DT_HOURS   # kWh por intervalo
const ETA_C = 0.95
const ETA_D = 0.95

# Pequeno custo de throughput para evitar ciclos artificiais
const CYCLE_COST = 0.01           # AUD/kWh movimentado

# Tarifa experimental
const PRICE_OFFPEAK = 0.20        # AUD/kWh
const PRICE_SHOULDER = 0.30
const PRICE_PEAK = 0.50
const PRICE_EXPORT = 0.08

# PV do Customer 12: referência ~1.04 kWp.
# Escalamos para um sistema residencial de 4 kWp.
const PV_SCALE = 4.0 / 1.04

# LDR
const NLAGS = 5
const POLY_DEGREE = 3

# Limiar para contar coeficientes não nulos
const NZ_TOL = 1e-5


# ============================================================
# Diretórios
# ============================================================

const DATA_DIR = joinpath(pwd(), "data_ausgrid")
const OUTPUT_DIR = joinpath(pwd(), "outputs_stor")

# Gráficos
const FIG_DIR = joinpath(OUTPUT_DIR, "graficos")
const FIG_NOTEBOOK_DIR = joinpath(FIG_DIR, "notebook")
const FIG_PRESENTATION_DIR = joinpath(FIG_DIR, "apresentacao")
const FIG_PRESENTATION_PNG_DIR = joinpath(FIG_PRESENTATION_DIR, "png")
const FIG_PRESENTATION_PDF_DIR = joinpath(FIG_PRESENTATION_DIR, "pdf")
const FIG_PRESENTATION_SVG_DIR = joinpath(FIG_PRESENTATION_DIR, "svg")

# CSVs
const CSV_DIR = joinpath(OUTPUT_DIR, "csv")
const CSV_RESULTS_DIR = joinpath(CSV_DIR, "resultados")
const CSV_POLICIES_DIR = joinpath(CSV_DIR, "politicas")
const CSV_PRESENTATION_DIR = joinpath(CSV_DIR, "apresentacao")

# Arquivos auxiliares
const LATEX_DIR = joinpath(OUTPUT_DIR, "latex")
const SUMMARY_DIR = joinpath(OUTPUT_DIR, "resumos")

for dir in [
    DATA_DIR,
    OUTPUT_DIR,
    FIG_DIR,
    FIG_NOTEBOOK_DIR,
    FIG_PRESENTATION_DIR,
    FIG_PRESENTATION_PNG_DIR,
    FIG_PRESENTATION_PDF_DIR,
    FIG_PRESENTATION_SVG_DIR,
    CSV_DIR,
    CSV_RESULTS_DIR,
    CSV_POLICIES_DIR,
    CSV_PRESENTATION_DIR,
    LATEX_DIR,
    SUMMARY_DIR
]
    mkpath(dir)
end

println("EMAX por intervalo = ", EMAX, " kWh")
println()
println("Estrutura de outputs criada:")
println("outputs_stor/")
println("├── graficos/notebook/")
println("├── graficos/apresentacao/png/")
println("├── graficos/apresentacao/pdf/")
println("├── graficos/apresentacao/svg/")
println("├── csv/resultados/")
println("├── csv/politicas/")
println("├── csv/apresentacao/")
println("├── latex/")
println("└── resumos/")


# ============================================================
# Cell 3
# ============================================================
# ============================================================
# 2. Download dos dados
# ============================================================

const DATA_URL = "https://raw.githubusercontent.com/pierre-haessig/solarhome-control-bench/master/data/data_2011-2012.csv"
const DATA_FILE = joinpath(DATA_DIR, "customer12_2011_2012.csv")

if !isfile(DATA_FILE)
    println("Baixando dados Ausgrid...")
    Downloads.download(DATA_URL, DATA_FILE)
    println("Download concluído: ", DATA_FILE)
else
    println("Arquivo já existe: ", DATA_FILE)
end

println("Tamanho do arquivo: ", round(filesize(DATA_FILE) / 1024^2, digits=2), " MB")

# ============================================================
# Cell 4
# ============================================================
# ============================================================
# 3. Leitura
# ============================================================

df = CSV.read(DATA_FILE, DataFrame)

# O CSV possui a data/hora na primeira coluna, originalmente sem nome.
rename!(df, names(df)[1] => :datetime)

df.datetime = [
    DateTime(String(x), dateformat"yyyy-mm-dd HH:MM:SS")
    for x in df.datetime
]

df.GC = Float64.(df.GC)
df.GG = Float64.(df.GG) .* PV_SCALE
df.net_load = df.GC .- df.GG
df.day = Date.(df.datetime)

sort!(df, :datetime)

println(first(df, 5))
println()
@printf("Observações: %d\n", nrow(df))
@printf("Período: %s até %s\n", minimum(df.datetime), maximum(df.datetime))
@printf("GC médio: %.3f kWh/30min\n", mean(df.GC))
@printf("GG médio escalado: %.3f kWh/30min\n", mean(df.GG))

# ============================================================
# Cell 5
# ============================================================
# ============================================================
# 4. Tarifa experimental por horário
# ============================================================

function import_tariff(dt::DateTime)
    h = hour(dt) + minute(dt) / 60

    if 15.5 <= h < 21.0
        return PRICE_PEAK
    elseif 10.5 <= h < 14.0
        return PRICE_OFFPEAK
    else
        return PRICE_SHOULDER
    end
end

df.p_import = import_tariff.(df.datetime)
df.p_export = fill(PRICE_EXPORT, nrow(df))

first(df, 8)

# ============================================================
# Cell 6
# ============================================================
# ============================================================
# 5. Estrutura de cenários
# ============================================================

struct DayScenarios
    dates::Vector{Date}
    load::Matrix{Float64}      # N x T, kWh
    pv::Matrix{Float64}        # N x T, kWh
    p_import::Matrix{Float64}  # N x T, AUD/kWh
    p_export::Matrix{Float64}  # N x T, AUD/kWh
end

function build_day_scenarios(df::DataFrame)
    groups = groupby(df, :day)

    valid_groups = DataFrame[]
    for g in groups
        if nrow(g) == 48
            gg = sort(DataFrame(g), :datetime)
            push!(valid_groups, gg)
        end
    end

    N = length(valid_groups)
    T = 48

    dates = Vector{Date}(undef, N)
    load = zeros(N, T)
    pv = zeros(N, T)
    p_import = zeros(N, T)
    p_export = zeros(N, T)

    for n in 1:N
        g = valid_groups[n]
        dates[n] = first(g.day)
        load[n, :] .= g.GC
        pv[n, :] .= g.GG
        p_import[n, :] .= g.p_import
        p_export[n, :] .= g.p_export
    end

    return DayScenarios(dates, load, pv, p_import, p_export)
end

all_days = build_day_scenarios(df)

@printf("Dias completos: %d\n", length(all_days.dates))
@printf("Estágios por dia: %d\n", size(all_days.load, 2))

# ============================================================
# Cell 7
# ============================================================
# ============================================================
# 6. Split cronológico 70/15/15
# ============================================================

function slice_scenarios(data::DayScenarios, idx)
    return DayScenarios(
        data.dates[idx],
        data.load[idx, :],
        data.pv[idx, :],
        data.p_import[idx, :],
        data.p_export[idx, :]
    )
end

Ndays = length(all_days.dates)

# Split cronológico:
# 70% treinamento
# 15% validação
# 15% teste OOS
n_train = floor(Int, 0.70 * Ndays)
n_val = floor(Int, 0.15 * Ndays)
n_test = Ndays - n_train - n_val

idx_train = 1:n_train
idx_val = (n_train + 1):(n_train + n_val)
idx_test = (n_train + n_val + 1):Ndays

train = slice_scenarios(all_days, idx_train)
val = slice_scenarios(all_days, idx_val)
test = slice_scenarios(all_days, idx_test)

println("==============================================")
println("SPLIT CRONOLÓGICO 70 / 15 / 15")
println("==============================================")

println(
    "Treinamento: ",
    first(train.dates), " -> ", last(train.dates),
    " | N = ", length(train.dates),
    " | ", round(100 * length(train.dates) / Ndays, digits=2), "%"
)

println(
    "Validação:   ",
    first(val.dates), " -> ", last(val.dates),
    " | N = ", length(val.dates),
    " | ", round(100 * length(val.dates) / Ndays, digits=2), "%"
)

println(
    "Teste OOS:   ",
    first(test.dates), " -> ", last(test.dates),
    " | N = ", length(test.dates),
    " | ", round(100 * length(test.dates) / Ndays, digits=2), "%"
)

println("Total de dias = ", Ndays)

@assert length(train.dates) + length(val.dates) + length(test.dates) == Ndays

# ============================================================
# Cell 8
# ============================================================
# ============================================================
# 7. Exemplo de um dia
# ============================================================

example_day = 1
hours = collect(0.0:0.5:23.5)

p_data = plot(
    hours,
    train.load[example_day, :],
    label="Consumo (GC)",
    xlabel="Hora",
    ylabel="kWh / 30 min",
    title="Ausgrid — exemplo de um dia",
    linewidth=2
)

plot!(
    p_data,
    hours,
    train.pv[example_day, :],
    label="Geração solar (GG escalada)",
    linewidth=2
)

savefig(p_data, joinpath(FIG_NOTEBOOK_DIR, "01_dados_exemplo.png"))
p_data

# ============================================================
# Cell 9
# ============================================================
# ============================================================
# 8. Features da LDR
# ============================================================

train_net = train.load .- train.pv
mu_net = mean(train_net)
sd_net = std(vec(train_net))

@printf("Média da carga líquida no treino: %.4f\n", mu_net)
@printf("Desvio-padrão: %.4f\n", sd_net)

function build_features(
    data::DayScenarios,
    mu_net::Float64,
    sd_net::Float64;
    nlags::Int=NLAGS,
    degree::Int=POLY_DEGREE
)
    N, T = size(data.load)
    J = 1 + (nlags + 1) * degree

    Phi = zeros(N, T, J)
    net = data.load .- data.pv
    z = (net .- mu_net) ./ sd_net

    for n in 1:N
        for t in 1:T
            Phi[n, t, 1] = 1.0
            j = 2

            for lag in 0:nlags
                tau = t - lag

                for k in 1:degree
                    if tau >= 1
                        Phi[n, t, j] = z[n, tau]^k
                    else
                        Phi[n, t, j] = 0.0
                    end
                    j += 1
                end
            end
        end
    end

    return Phi
end

Phi_train = build_features(train, mu_net, sd_net)
Phi_val = build_features(val, mu_net, sd_net)
Phi_test = build_features(test, mu_net, sd_net)

println("Número de features por estágio = ", size(Phi_train, 3))
println("Número total de coeficientes theta = ",
        size(Phi_train, 2) * size(Phi_train, 3))

# ============================================================
# Cell 10
# ============================================================
# ============================================================
# 9. Baseline sem bateria
# ============================================================

function no_battery_costs(data::DayScenarios)
    N, T = size(data.load)
    costs = zeros(N)

    for n in 1:N
        for t in 1:T
            net = data.load[n, t] - data.pv[n, t]

            if net >= 0
                costs[n] += data.p_import[n, t] * net
            else
                costs[n] -= data.p_export[n, t] * (-net)
            end
        end
    end

    return costs
end

baseline_train = no_battery_costs(train)
baseline_val = no_battery_costs(val)
baseline_test = no_battery_costs(test)

@printf("Custo médio sem bateria — treino: AUD %.3f/dia\n", mean(baseline_train))
@printf("Custo médio sem bateria — validação: AUD %.3f/dia\n", mean(baseline_val))
@printf("Custo médio sem bateria — teste: AUD %.3f/dia\n", mean(baseline_test))

# ============================================================
# Cell 11
# ============================================================
# ============================================================
# 10. Ajuste LDR / AdaLASSO
# ============================================================

function fit_ldr(
    data::DayScenarios,
    Phi::Array{Float64,3};
    lambda::Float64=0.0,
    weights=nothing,
    output_flag::Int=0
)
    N, T = size(data.load)
    J = size(Phi, 3)

    if weights === nothing
        weights = ones(T, J)
        weights[:, 1] .= 0.0
    end

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", output_flag)

    @variable(model, 0 <= B[1:N, 1:T] <= BMAX)
    @variable(model, 0 <= charge[1:N, 1:T] <= EMAX)
    @variable(model, 0 <= discharge[1:N, 1:T] <= EMAX)
    @variable(model, grid_import[1:N, 1:T] >= 0)
    @variable(model, grid_export[1:N, 1:T] >= 0)

    @variable(model, theta[1:T, 1:J])
    @variable(model, abs_theta[1:T, 2:J] >= 0)

    # Dinâmica da bateria
    for n in 1:N
        @constraint(
            model,
            B[n, 1] ==
            B0 + ETA_C * charge[n, 1] - discharge[n, 1] / ETA_D
        )

        for t in 2:T
            @constraint(
                model,
                B[n, t] ==
                B[n, t-1] +
                ETA_C * charge[n, t] -
                discharge[n, t] / ETA_D
            )
        end

        # Evita drenar a bateria no final do dia
        @constraint(model, B[n, T] >= B0)
    end

    # Balanço de energia e regra de decisão
    for n in 1:N
        for t in 1:T
            @constraint(
                model,
                grid_import[n, t] +
                data.pv[n, t] +
                discharge[n, t]
                ==
                data.load[n, t] +
                charge[n, t] +
                grid_export[n, t]
            )

            @constraint(
                model,
                B[n, t] ==
                sum(theta[t, j] * Phi[n, t, j] for j in 1:J)
            )
        end
    end

    # Linearização da norma L1
    for t in 1:T
        for j in 2:J
            @constraint(model, abs_theta[t, j] >= theta[t, j])
            @constraint(model, abs_theta[t, j] >= -theta[t, j])
        end
    end

    operating_cost =
        (1 / N) *
        sum(
            data.p_import[n, t] * grid_import[n, t]
            - data.p_export[n, t] * grid_export[n, t]
            + CYCLE_COST * (charge[n, t] + discharge[n, t])
            for n in 1:N, t in 1:T
        )

    # Normalizamos pelo número de coeficientes penalizados.
    # Isso só muda a escala de lambda, não a família de políticas.
    penalty =
        sum(
            weights[t, j] * abs_theta[t, j]
            for t in 1:T, j in 2:J
        ) / (T * (J - 1))

    # OBJETIVO DA LDR / AdaLASSO
    # lambda = 0  -> LDR clássica
    # lambda > 0  -> LDR regularizada por AdaLASSO
    @objective(
        model,
        Min,
        operating_cost + lambda * penalty
    )

    elapsed = @elapsed optimize!(model)

    status = termination_status(model)

    if status != MOI.OPTIMAL
        error("O ajuste terminou com status: $status")
    end

    theta_hat = [
        value(theta[t, j])
        for t in 1:T, j in 1:J
    ]

    return (
        theta=theta_hat,
        objective=objective_value(model),
        train_operating_cost=value(operating_cost),
        penalty=value(penalty),
        time=elapsed,
        status=status
    )
end

println("Função fit_ldr definida.")

# ============================================================
# Cell 12
# ============================================================
# ============================================================
# 11. Ajuste da LDR não regularizada
# ============================================================

fit0 = fit_ldr(
    train,
    Phi_train;
    lambda=0.0,
    output_flag=0
)

@printf("Status: %s\n", fit0.status)
@printf("Custo operacional de treino: AUD %.4f/dia\n", fit0.train_operating_cost)
@printf("Tempo: %.2f s\n", fit0.time)
@printf("Coeficientes não nulos: %d\n",
        count(abs.(fit0.theta[:, 2:end]) .> NZ_TOL))

# ============================================================
# Cell 13
# ============================================================
# ============================================================
# 12. Pesos AdaLASSO
# ============================================================

function adalasso_weights(theta0::Matrix{Float64}; eps_zero=1e-6)
    T, J = size(theta0)
    w = zeros(T, J)

    for t in 1:T
        w[t, 1] = 0.0  # intercepto não penalizado

        for j in 2:J
            a = abs(theta0[t, j])
            denom = a > eps_zero ? a : 1.0
            w[t, j] = 1.0 / denom
        end
    end

    return w
end

ada_weights = adalasso_weights(fit0.theta)

println("Pesos AdaLASSO calculados.")

# ============================================================
# Cell 14
# ============================================================
# ============================================================
# 13. Avaliação OOS por state-target tracking
# ============================================================

function evaluate_policy(
    data::DayScenarios,
    Phi::Array{Float64,3},
    theta::Matrix{Float64}
)
    N, T = size(data.load)
    J = size(Phi, 3)

    daily_cost = zeros(N)
    daily_import = zeros(N)
    daily_export = zeros(N)
    daily_tracking_error = zeros(N)

    B_path = zeros(N, T)
    charge_path = zeros(N, T)
    discharge_path = zeros(N, T)

    for n in 1:N
        Bprev = B0

        for t in 1:T
            target = sum(theta[t, j] * Phi[n, t, j] for j in 1:J)

            # Faixa fisicamente alcançável no próximo passo
            bmin_dyn = max(0.0, Bprev - EMAX / ETA_D)
            bmax_dyn = min(BMAX, Bprev + ETA_C * EMAX)

            # Viabilidade do estado terminal B_T >= B0:
            # após t ainda restam T-t oportunidades de carregar.
            terminal_floor = max(
                0.0,
                B0 - (T - t) * ETA_C * EMAX
            )

            bmin = max(bmin_dyn, terminal_floor)
            bmax = bmax_dyn

            if bmin > bmax + 1e-9
                error("Faixa de bateria inviável no cenário $n, estágio $t")
            end

            Bnow = clamp(target, bmin, bmax)

            if Bnow >= Bprev
                charge = (Bnow - Bprev) / ETA_C
                discharge = 0.0
            else
                charge = 0.0
                discharge = (Bprev - Bnow) * ETA_D
            end

            net_grid =
                data.load[n, t] +
                charge -
                data.pv[n, t] -
                discharge

            if net_grid >= 0
                gimp = net_grid
                gexp = 0.0
            else
                gimp = 0.0
                gexp = -net_grid
            end

            cost =
                data.p_import[n, t] * gimp -
                data.p_export[n, t] * gexp +
                CYCLE_COST * (charge + discharge)

            daily_cost[n] += cost
            daily_import[n] += gimp
            daily_export[n] += gexp
            daily_tracking_error[n] += abs(Bnow - target)

            B_path[n, t] = Bnow
            charge_path[n, t] = charge
            discharge_path[n, t] = discharge

            Bprev = Bnow
        end
    end

    return (
        costs=daily_cost,
        imports=daily_import,
        exports=daily_export,
        tracking_error=daily_tracking_error,
        B=B_path,
        charge=charge_path,
        discharge=discharge_path
    )
end

val0 = evaluate_policy(val, Phi_val, fit0.theta)

@printf("LDR não regularizada — custo médio validação: AUD %.4f/dia\n",
        mean(val0.costs))

# ============================================================
# Cell 15
# ============================================================
# ============================================================
# 14. Grid search do AdaLASSO
# ============================================================

cost_scale = mean(baseline_train)

lambda_rel_grid = [
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0
]

fits = Any[fit0]
lambda_abs = Float64[0.0]
lambda_rel = Float64[0.0]

val_mean_cost = Float64[mean(val0.costs)]
val_p95_cost = Float64[quantile(val0.costs, 0.95)]
nonzero = Int[count(abs.(fit0.theta[:, 2:end]) .> NZ_TOL)]
fit_time = Float64[fit0.time]

for r in lambda_rel_grid
    lam = r * cost_scale

    println("Ajustando lambda_rel = ", r,
            " | lambda_abs = ", round(lam, digits=6))

    fit = fit_ldr(
        train,
        Phi_train;
        lambda=lam,
        weights=ada_weights,
        output_flag=0
    )

    ev = evaluate_policy(val, Phi_val, fit.theta)

    push!(fits, fit)
    push!(lambda_abs, lam)
    push!(lambda_rel, r)
    push!(val_mean_cost, mean(ev.costs))
    push!(val_p95_cost, quantile(ev.costs, 0.95))
    push!(nonzero, count(abs.(fit.theta[:, 2:end]) .> NZ_TOL))
    push!(fit_time, fit.time)
end

validation_results = DataFrame(
    lambda_rel=lambda_rel,
    lambda_abs=lambda_abs,
    val_mean_cost=val_mean_cost,
    val_p95_cost=val_p95_cost,
    nonzero=nonzero,
    fit_time_s=fit_time
)

sort!(validation_results, :val_mean_cost)

validation_results

# ============================================================
# Cell 16
# ============================================================
# ============================================================
# 15. Escolha de lambda*
# ============================================================

best_idx = argmin(val_mean_cost)
best_fit = fits[best_idx]
best_lambda_rel = lambda_rel[best_idx]
best_lambda_abs = lambda_abs[best_idx]

println("==============================================")
println("Lambda selecionado")
println("lambda_rel = ", best_lambda_rel)
println("lambda_abs = ", best_lambda_abs)
@printf("Custo médio de validação = AUD %.4f/dia\n", val_mean_cost[best_idx])
println("Coeficientes não nulos = ", nonzero[best_idx])
println("==============================================")

# CHECK REGULARIZATION
if length(unique(round.(val_mean_cost, digits=8))) == 1 &&
   length(unique(nonzero)) == 1
    @warn """
    Todos os valores de lambda produziram exatamente o mesmo custo e o
    mesmo número de coeficientes. Isso pode indicar que a grade de lambda
    ainda é insuficiente ou que a penalização não está alterando a solução.
    Revise a escala de lambda antes de interpretar o AdaLASSO.
    """
end


# ============================================================
# Cell 17
# ============================================================
# ============================================================
# 16. Teste OOS
# ============================================================

test_ldr = evaluate_policy(test, Phi_test, fit0.theta)
test_ada = evaluate_policy(test, Phi_test, best_fit.theta)

function solve_oracle_day(
    load::Vector{Float64},
    pv::Vector{Float64},
    p_import::Vector{Float64},
    p_export::Vector{Float64};
    output_flag=0
)
    T = length(load)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", output_flag)

    @variable(model, 0 <= B[1:T] <= BMAX)
    @variable(model, 0 <= charge[1:T] <= EMAX)
    @variable(model, 0 <= discharge[1:T] <= EMAX)
    @variable(model, grid_import[1:T] >= 0)
    @variable(model, grid_export[1:T] >= 0)

    @constraint(
        model,
        B[1] == B0 + ETA_C * charge[1] - discharge[1] / ETA_D
    )

    for t in 2:T
        @constraint(
            model,
            B[t] ==
            B[t-1] +
            ETA_C * charge[t] -
            discharge[t] / ETA_D
        )
    end

    for t in 1:T
        @constraint(
            model,
            grid_import[t] + pv[t] + discharge[t]
            ==
            load[t] + charge[t] + grid_export[t]
        )
    end

    @constraint(model, B[T] >= B0)

    @objective(
        model,
        Min,
        sum(
            p_import[t] * grid_import[t]
            - p_export[t] * grid_export[t]
            + CYCLE_COST * (charge[t] + discharge[t])
            for t in 1:T
        )
    )

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        error("Oracle não ótimo: ", termination_status(model))
    end

    return (
        cost=objective_value(model),
        B=value.(B)
    )
end

Ntest, T = size(test.load)
oracle_cost = zeros(Ntest)
oracle_B = zeros(Ntest, T)

for n in 1:Ntest
    ans = solve_oracle_day(
        vec(test.load[n, :]),
        vec(test.pv[n, :]),
        vec(test.p_import[n, :]),
        vec(test.p_export[n, :])
    )
    oracle_cost[n] = ans.cost
    oracle_B[n, :] .= ans.B
end

println("Teste concluído.")

# ============================================================
# Cell 18
# ============================================================
# ============================================================
# 17. Tabela final
# ============================================================

function summary_row(method, costs, imports, exports, nnz, time_s)
    return (
        method=method,
        mean_cost=mean(costs),
        median_cost=median(costs),
        p95_cost=quantile(costs, 0.95),
        mean_import=mean(imports),
        mean_export=mean(exports),
        nonzero_coefficients=nnz,
        fit_time_s=time_s
    )
end

zero_vec = zeros(length(baseline_test))

results = DataFrame([
    summary_row(
        "Sem bateria",
        baseline_test,
        [sum(max.(test.load[n, :] .- test.pv[n, :], 0.0)) for n in 1:Ntest],
        [sum(max.(test.pv[n, :] .- test.load[n, :], 0.0)) for n in 1:Ntest],
        0,
        0.0
    ),

    summary_row(
        "LDR",
        test_ldr.costs,
        test_ldr.imports,
        test_ldr.exports,
        count(abs.(fit0.theta[:, 2:end]) .> NZ_TOL),
        fit0.time
    ),

    summary_row(
        "AdaLASSO-LDR",
        test_ada.costs,
        test_ada.imports,
        test_ada.exports,
        count(abs.(best_fit.theta[:, 2:end]) .> NZ_TOL),
        best_fit.time
    ),

    summary_row(
        "Oracle",
        oracle_cost,
        zero_vec,
        zero_vec,
        0,
        NaN
    )
])

results.cost_saving_vs_no_battery_pct =
    100 .* (
        mean(baseline_test) .- results.mean_cost
    ) ./ mean(baseline_test)

results

# ============================================================
# Cell 19
# ============================================================
# ============================================================
# 18. Diferença LDR x AdaLASSO
# ============================================================

delta_cost =
    100 * (mean(test_ldr.costs) - mean(test_ada.costs)) /
    mean(test_ldr.costs)

nnz_ldr = count(abs.(fit0.theta[:, 2:end]) .> NZ_TOL)
nnz_ada = count(abs.(best_fit.theta[:, 2:end]) .> NZ_TOL)

delta_nnz =
    100 * (nnz_ldr - nnz_ada) / max(nnz_ldr, 1)

println("==============================================")
@printf("LDR OOS:       AUD %.4f/dia\n", mean(test_ldr.costs))
@printf("AdaLASSO OOS:  AUD %.4f/dia\n", mean(test_ada.costs))
@printf("Variação de custo LDR -> AdaLASSO: %.2f%%\n", delta_cost)
println()
println("Coeficientes não nulos LDR:      ", nnz_ldr)
println("Coeficientes não nulos AdaLASSO: ", nnz_ada)
@printf("Redução de coeficientes: %.2f%%\n", delta_nnz)
println("==============================================")

# Métricas de parcimônia, análogas às reportadas por Nazare & Street
n_penalized = length(fit0.theta[:, 2:end])

nnz_pct_ldr = 100 * nnz_ldr / n_penalized
nnz_pct_ada = 100 * nnz_ada / n_penalized

l1_ldr = sum(abs.(fit0.theta[:, 2:end]))
l1_ada = sum(abs.(best_fit.theta[:, 2:end]))

l1_shrink_pct =
    100 * (l1_ldr - l1_ada) / max(l1_ldr, eps())

@printf("Percentual não nulo LDR: %.2f%%\n", nnz_pct_ldr)
@printf("Percentual não nulo AdaLASSO: %.2f%%\n", nnz_pct_ada)
@printf("Redução da norma L1: %.2f%%\n", l1_shrink_pct)


# ============================================================
# Cell 20
# ============================================================
# ============================================================
# 19. Gráfico 1 — custo de validação x lambda
# ============================================================

plot_df = DataFrame(
    lambda_rel=lambda_rel,
    cost=val_mean_cost,
    nonzero=nonzero
)

xpos = collect(1:nrow(plot_df))
labels = [
    x == 0.0 ? "0" : @sprintf("%.0e", x)
    for x in plot_df.lambda_rel
]

p_lambda = plot(
    xpos,
    plot_df.cost,
    marker=:circle,
    linewidth=2,
    xticks=(xpos, labels),
    xrotation=45,
    xlabel="lambda relativo",
    ylabel="Custo médio de validação (AUD/dia)",
    title="Seleção de regularização",
    label="Validação"
)

savefig(
    p_lambda,
    joinpath(FIG_NOTEBOOK_DIR, "02_validacao_lambda.png")
)

p_lambda

# ============================================================
# Cell 21
# ============================================================
# ============================================================
# 20. Gráfico 2 — parcimônia x lambda
# ============================================================

p_sparse = plot(
    xpos,
    plot_df.nonzero,
    marker=:circle,
    linewidth=2,
    xticks=(xpos, labels),
    xrotation=45,
    xlabel="lambda relativo",
    ylabel="Coeficientes não nulos",
    title="Parcimônia da política",
    label="nnz(theta)"
)

savefig(
    p_sparse,
    joinpath(FIG_NOTEBOOK_DIR, "03_coeficientes_lambda.png")
)

p_sparse

# ============================================================
# Cell 22
# ============================================================
# ============================================================
# 21. Gráfico 3 — distribuição de custos OOS
# ============================================================

cost_df = DataFrame(
    method=vcat(
        fill("Sem bateria", length(baseline_test)),
        fill("LDR", length(test_ldr.costs)),
        fill("AdaLASSO", length(test_ada.costs)),
        fill("Oracle", length(oracle_cost))
    ),
    cost=vcat(
        baseline_test,
        test_ldr.costs,
        test_ada.costs,
        oracle_cost
    )
)

p_box = @df cost_df boxplot(
    :method,
    :cost,
    legend=false,
    xlabel="Método",
    ylabel="Custo diário OOS (AUD)",
    title="Distribuição dos custos no teste"
)

savefig(
    p_box,
    joinpath(FIG_NOTEBOOK_DIR, "04_boxplot_custos_oos.png")
)

p_box

# ============================================================
# Cell 23
# ============================================================
# ============================================================
# 22. Gráfico 4 — trajetória da bateria em um dia de teste
# ============================================================

day_to_plot = 1

p_battery = plot(
    hours,
    test_ldr.B[day_to_plot, :],
    linewidth=2,
    label="LDR",
    xlabel="Hora",
    ylabel="Estado de carga (kWh)",
    title="Trajetória da bateria — dia de teste"
)

plot!(
    p_battery,
    hours,
    test_ada.B[day_to_plot, :],
    linewidth=2,
    label="AdaLASSO"
)

plot!(
    p_battery,
    hours,
    oracle_B[day_to_plot, :],
    linewidth=2,
    linestyle=:dash,
    label="Oracle"
)

savefig(
    p_battery,
    joinpath(FIG_NOTEBOOK_DIR, "05_bateria_dia_teste.png")
)

p_battery

# ============================================================
# Cell 24
# ============================================================
# ============================================================
# 23. Exportar resultados em pastas separadas
# ============================================================

# Resultados gerais
CSV.write(
    joinpath(CSV_RESULTS_DIR, "validation_results.csv"),
    validation_results
)

CSV.write(
    joinpath(CSV_RESULTS_DIR, "test_results.csv"),
    results
)

# Coeficientes / políticas
theta_ldr_df = DataFrame(
    fit0.theta,
    :auto
)

theta_ada_df = DataFrame(
    best_fit.theta,
    :auto
)

CSV.write(
    joinpath(CSV_POLICIES_DIR, "theta_ldr.csv"),
    theta_ldr_df
)

CSV.write(
    joinpath(CSV_POLICIES_DIR, "theta_adalasso.csv"),
    theta_ada_df
)

println("Resultados gerais em:")
println(CSV_RESULTS_DIR)
println()

println("Políticas estimadas em:")
println(CSV_POLICIES_DIR)

# ============================================================
# Cell 25
# ============================================================
# ============================================================
# 24. EXPORTAÇÃO FINAL ORGANIZADA PARA A APRESENTAÇÃO
# ============================================================

println("==================================================")
println("EXPORTANDO MATERIAL DA APRESENTAÇÃO")
println("==================================================")
println()

# ------------------------------------------------------------
# Função: salvar cada gráfico final em PNG, PDF e SVG
# ------------------------------------------------------------

function save_slide_figure(p, basename::String)

    plot!(
        p;
        size=(1600, 900),
        dpi=300,
        background_color=:white,
        foreground_color=:black
    )

    savefig(
        p,
        joinpath(
            FIG_PRESENTATION_PNG_DIR,
            basename * ".png"
        )
    )

    savefig(
        p,
        joinpath(
            FIG_PRESENTATION_PDF_DIR,
            basename * ".pdf"
        )
    )

    savefig(
        p,
        joinpath(
            FIG_PRESENTATION_SVG_DIR,
            basename * ".svg"
        )
    )

    println("✓ ", basename)
end

# ------------------------------------------------------------
# 1. Gráficos já construídos no notebook
# ------------------------------------------------------------

save_slide_figure(
    p_data,
    "fig_01_ausgrid_dia_exemplo"
)

save_slide_figure(
    p_lambda,
    "fig_02_validacao_lambda"
)

save_slide_figure(
    p_sparse,
    "fig_03_parcimonia_coeficientes"
)

save_slide_figure(
    p_box,
    "fig_04_distribuicao_custos_oos"
)

save_slide_figure(
    p_battery,
    "fig_05_trajetoria_bateria"
)

# ------------------------------------------------------------
# 2. Treino vs validação
# ------------------------------------------------------------

train_cost_by_lambda = [
    fit.train_operating_cost
    for fit in fits
]

p_train_val = plot(
    xpos,
    train_cost_by_lambda;
    marker=:circle,
    linewidth=2.5,
    label="Treino (in-sample)",
    xticks=(xpos, labels),
    xrotation=45,
    xlabel="lambda relativo",
    ylabel="Custo médio (AUD/dia)",
    title="Regularização: treino vs. validação",
    legend=:best
)

plot!(
    p_train_val,
    xpos,
    val_mean_cost;
    marker=:diamond,
    linewidth=2.5,
    label="Validação"
)

vline!(
    p_train_val,
    [best_idx];
    linestyle=:dash,
    linewidth=1.7,
    label="lambda*"
)

save_slide_figure(
    p_train_val,
    "fig_06_treino_vs_validacao"
)

# ------------------------------------------------------------
# 3. Custo médio OOS por método
# ------------------------------------------------------------

methods_plot = [
    "Sem bateria",
    "LDR",
    "AdaLASSO",
    "Oracle"
]

mean_costs_plot = [
    mean(baseline_test),
    mean(test_ldr.costs),
    mean(test_ada.costs),
    mean(oracle_cost)
]

p_mean_cost = bar(
    methods_plot,
    mean_costs_plot;
    legend=false,
    ylabel="Custo médio OOS (AUD/dia)",
    xlabel="Método",
    title="Custo médio fora da amostra",
    xrotation=15
)

save_slide_figure(
    p_mean_cost,
    "fig_07_custo_medio_oos"
)

# ------------------------------------------------------------
# 4. Tabela resumida dos métodos
# ------------------------------------------------------------

presentation_results = DataFrame(
    Metodo = [
        "Sem bateria",
        "LDR",
        "AdaLASSO-LDR",
        "Oracle"
    ],

    Custo_medio_OOS_AUD_dia = [
        mean(baseline_test),
        mean(test_ldr.costs),
        mean(test_ada.costs),
        mean(oracle_cost)
    ],

    P95_custo_AUD_dia = [
        quantile(baseline_test, 0.95),
        quantile(test_ldr.costs, 0.95),
        quantile(test_ada.costs, 0.95),
        quantile(oracle_cost, 0.95)
    ],

    Coeficientes_nao_nulos = [
        0,
        nnz_ldr,
        nnz_ada,
        0
    ],

    Tempo_ajuste_s = [
        0.0,
        fit0.time,
        best_fit.time,
        NaN
    ]
)

baseline_mean = mean(baseline_test)

presentation_results.Economia_vs_sem_bateria_pct =
    100 .* (
        baseline_mean .-
        presentation_results.Custo_medio_OOS_AUD_dia
    ) ./ baseline_mean

CSV.write(
    joinpath(
        CSV_PRESENTATION_DIR,
        "table_01_resultados_finais.csv"
    ),
    presentation_results
)

# ------------------------------------------------------------
# 5. Resultados do grid de lambda
# ------------------------------------------------------------

validation_slide = DataFrame(
    lambda_rel = lambda_rel,
    lambda_abs = lambda_abs,
    custo_treino = train_cost_by_lambda,
    custo_validacao = val_mean_cost,
    p95_validacao = val_p95_cost,
    coeficientes_nao_nulos = nonzero,
    tempo_ajuste_s = fit_time
)

CSV.write(
    joinpath(
        CSV_PRESENTATION_DIR,
        "table_02_validacao_lambda.csv"
    ),
    validation_slide
)

# ------------------------------------------------------------
# 6. Custos individuais OOS
# ------------------------------------------------------------

oos_long = DataFrame(
    Metodo = vcat(
        fill("Sem bateria", length(baseline_test)),
        fill("LDR", length(test_ldr.costs)),
        fill("AdaLASSO-LDR", length(test_ada.costs)),
        fill("Oracle", length(oracle_cost))
    ),

    Custo_AUD_dia = vcat(
        baseline_test,
        test_ldr.costs,
        test_ada.costs,
        oracle_cost
    )
)

CSV.write(
    joinpath(
        CSV_PRESENTATION_DIR,
        "data_01_custos_oos_long.csv"
    ),
    oos_long
)

# ------------------------------------------------------------
# 7. Dados da trajetória operacional
# ------------------------------------------------------------

battery_example = DataFrame(
    Hora = hours,
    LDR_kWh = vec(test_ldr.B[day_to_plot, :]),
    AdaLASSO_kWh = vec(test_ada.B[day_to_plot, :]),
    Oracle_kWh = vec(oracle_B[day_to_plot, :]),
    Consumo_kWh = vec(test.load[day_to_plot, :]),
    Solar_kWh = vec(test.pv[day_to_plot, :])
)

CSV.write(
    joinpath(
        CSV_PRESENTATION_DIR,
        "data_02_bateria_dia_exemplo.csv"
    ),
    battery_example
)

# ------------------------------------------------------------
# 8. Indicadores finais
# ------------------------------------------------------------

saving_ada_vs_baseline =
    100 * (
        mean(baseline_test) -
        mean(test_ada.costs)
    ) / mean(baseline_test)

gap_ada_vs_oracle =
    100 * (
        mean(test_ada.costs) -
        mean(oracle_cost)
    ) / abs(mean(oracle_cost))

# ------------------------------------------------------------
# 9. Resumo textual
# ------------------------------------------------------------

summary_txt = joinpath(
    SUMMARY_DIR,
    "presentation_summary.txt"
)

open(summary_txt, "w") do io

    println(io, "IND2097 - Regularized Linear Decision Rules")
    println(io, "Ausgrid residential battery experiment")
    println(io, "==================================================")
    println(io)

    println(io, "DADOS")
    println(io, "Treino: $(first(train.dates)) -> $(last(train.dates))")
    println(io, "Validação: $(first(val.dates)) -> $(last(val.dates))")
    println(io, "Teste OOS: $(first(test.dates)) -> $(last(test.dates))")
    println(io)

    println(io, "REGULARIZAÇÃO")
    println(io, "lambda_rel* = $(best_lambda_rel)")
    println(io, "lambda_abs* = $(best_lambda_abs)")
    println(io, "nnz LDR = $(nnz_ldr)")
    println(io, "nnz AdaLASSO = $(nnz_ada)")
    println(io, @sprintf(
        "Redução nnz = %.2f%%",
        delta_nnz
    ))
    println(io)

    println(io, "RESULTADOS OOS")
    println(io, @sprintf(
        "Sem bateria = AUD %.4f/dia",
        mean(baseline_test)
    ))
    println(io, @sprintf(
        "LDR = AUD %.4f/dia",
        mean(test_ldr.costs)
    ))
    println(io, @sprintf(
        "AdaLASSO = AUD %.4f/dia",
        mean(test_ada.costs)
    ))
    println(io, @sprintf(
        "Oracle = AUD %.4f/dia",
        mean(oracle_cost)
    ))
    println(io)

    println(io, @sprintf(
        "Redução LDR -> AdaLASSO = %.2f%%",
        delta_cost
    ))

    println(io, @sprintf(
        "Economia AdaLASSO vs sem bateria = %.2f%%",
        saving_ada_vs_baseline
    ))

    println(io, @sprintf(
        "Gap AdaLASSO vs oracle = %.2f%%",
        gap_ada_vs_oracle
    ))
end

# ------------------------------------------------------------
# 10. Macros LaTeX para o Beamer
# ------------------------------------------------------------

tex_file = joinpath(
    LATEX_DIR,
    "presentation_values.tex"
)

open(tex_file, "w") do io

    println(io, "% Gerado automaticamente pelo notebook")
    println(io)

    println(
        io,
        "\\newcommand{\\BestLambdaRel}{",
        @sprintf("%.4g", best_lambda_rel),
        "}"
    )

    println(
        io,
        "\\newcommand{\\BestLambdaAbs}{",
        @sprintf("%.4g", best_lambda_abs),
        "}"
    )

    println(
        io,
        "\\newcommand{\\CostNoBattery}{",
        @sprintf("%.2f", mean(baseline_test)),
        "}"
    )

    println(
        io,
        "\\newcommand{\\CostLDR}{",
        @sprintf("%.2f", mean(test_ldr.costs)),
        "}"
    )

    println(
        io,
        "\\newcommand{\\CostAda}{",
        @sprintf("%.2f", mean(test_ada.costs)),
        "}"
    )

    println(
        io,
        "\\newcommand{\\CostOracle}{",
        @sprintf("%.2f", mean(oracle_cost)),
        "}"
    )

    println(
        io,
        "\\newcommand{\\CostReductionAda}{",
        @sprintf("%.2f", delta_cost),
        "\\%}"
    )

    println(
        io,
        "\\newcommand{\\SavingsAdaBaseline}{",
        @sprintf("%.2f", saving_ada_vs_baseline),
        "\\%}"
    )

    println(
        io,
        "\\newcommand{\\NNZLDR}{",
        nnz_ldr,
        "}"
    )

    println(
        io,
        "\\newcommand{\\NNZAda}{",
        nnz_ada,
        "}"
    )

    println(
        io,
        "\\newcommand{\\NNZReduction}{",
        @sprintf("%.2f", delta_nnz),
        "\\%}"
    )
end

# ------------------------------------------------------------
# 11. README dos outputs
# ------------------------------------------------------------

readme_file = joinpath(
    SUMMARY_DIR,
    "README_OUTPUTS.txt"
)

open(readme_file, "w") do io

    println(io, "OUTPUTS DO EXPERIMENTO")
    println(io, "==============================================")
    println(io)

    println(io, "GRAFICOS PARA APRESENTACAO:")
    println(io, "  graficos/apresentacao/png/")
    println(io, "  graficos/apresentacao/pdf/")
    println(io, "  graficos/apresentacao/svg/")
    println(io)

    println(io, "CSVS GERAIS:")
    println(io, "  csv/resultados/")
    println(io)

    println(io, "COEFICIENTES DAS POLITICAS:")
    println(io, "  csv/politicas/")
    println(io)

    println(io, "CSVS PARA SLIDES:")
    println(io, "  csv/apresentacao/")
    println(io)

    println(io, "MACROS LATEX:")
    println(io, "  latex/presentation_values.tex")
    println(io)

    println(io, "RESUMO:")
    println(io, "  resumos/presentation_summary.txt")
end

# ------------------------------------------------------------
# 12. Listagem final
# ------------------------------------------------------------

println()
println("==================================================")
println("EXPORTAÇÃO CONCLUÍDA")
println("==================================================")
println()

println("GRÁFICOS PNG PARA OVERLEAF:")
println(FIG_PRESENTATION_PNG_DIR)

for file in sort(readdir(FIG_PRESENTATION_PNG_DIR))
    println("  ", file)
end

println()
println("CSVs PARA A APRESENTAÇÃO:")
println(CSV_PRESENTATION_DIR)

for file in sort(readdir(CSV_PRESENTATION_DIR))
    println("  ", file)
end

println()
println("Arquivo LaTeX:")
println(joinpath(LATEX_DIR, "presentation_values.tex"))