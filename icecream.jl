using CSV
using DataFrames
using Statistics
using Dates

path = "data/IPN31152N.csv"
df_ic = CSV.read(path, DataFrame)

path = "data/Preventable-injury-related-deaths-by-month-and-cause.csv"
df_d = CSV.read(path, DataFrame)

proper_col_names = (collect ∘ skipmissing ∘ collect)(df_d[1, :])
rename!(df_d, 1:length(proper_col_names) .=> Symbol.(proper_col_names))
df_d = df_d[3:end, :]

df_d = df_d[:, mean.(ismissing, eachcol(df_d)) .< 0.1] # Drop columns with 90% or more missing
filter!(:Month => !=(""), df_d)
df_d[!, ["Year", "Month", "Drowning (b)"]]


df_dm = filter(:Month => !=("Total"), df_d)[!, ["Year", "Month", "Drowning (b)"]]

dm_years = parse.(Int64, df_dm[:, "Year"])
dm_months = replace(df_dm[:, "Month"], Dates.LOCALES["english"].month_value...)
dm_days = ones(Int64, length(dm_years))
df_dm[!, "DATE"] = Date.(dm_years, dm_months, dm_days)


# Plot catplots by month

using CairoMakie
set_theme!(theme_black())

df_dm[!, "Drowning"] = parse.(Float64, df_dm[!, "Drowning (b)"])
drowning_month_avg = combine(groupby(df_dm, :Month), :Drowning => mean)

df_ic[!, "Month"] = month.(df_ic[!, :DATE])
icecream_month_avg = combine(groupby(df_ic, :Month), :IPN31152N => mean)

figs = []
for (title, yvals) in zip(["Average Number of Drowning Deaths by Month", "Average Production of Ice Cream and Frozen Desert by Month"], 
    [drowning_month_avg[!, "Drowning_mean"], icecream_month_avg[!, "IPN31152N_mean"]])
    fig, _, _ = barplot(1:12, yvals, color = :tomato,
        axis = (; xticks = (1:12, unique(df_dm[!, :Month])), xticklabelrotation=45.0, 
            title = title, limits = (nothing, nothing, 0, nothing),
            xgridvisible = false, topspinevisible = false, rightspinevisible = false, leftspinevisible = false)
    )
    push!(figs, fig)
end
figs[1]
figs[2]

# ----------------------------

df = innerjoin(df_dm[!, Not([:Drowning])], df_ic[!, Not([:Month])], on=:DATE)
select!(df, Not([:Year, :Month]))
rename!(df, Symbol("Drowning (b)") => "drowning", :IPN31152N => "sales")
df[!, "drowning"] = parse.(Float64, df[!, "drowning"])
df

# Plot Drowning vs. Ice cream sales

set_theme!(theme_black())
fig, ax, scat = scatter(df[!, "sales"], df[!, "drowning"],
    axis=(; title = "Ice cream sales vs. Drowning",
        xlabel = "Industrial Production of Ice Cream and Frozen Desert (% of 2017)",
        ylabel = "Deaths by Drowning per Month, Excluding Water Transport Drownings"))

# Perform a linear regression

using MLJ

X = df[:, ["sales"]]
y = df[:, "drowning"]

LinearModel = @load LinearRegressor pkg=GLM
model = LinearModel()
evaluate(model, X, y; resampling = CV(shuffle=true), measures = [root_mean_squared_error], verbosity = 0)

mach = machine(model, X, y)
fit!(mach)
report(mach)

coef = DataFrame(report(mach).coef_table)[!, "Coef."]
f(x) = coef[1] * x + coef[2]
f.(X[!, "sales"])

xlims!(ax, ax.xaxis.attributes.limits[])
ylims!(ax, ax.yaxis.attributes.limits[])

lines!(ax, [-100, 300], f.([-100, 300]); color = :orange, linestyle = :dash)
fig

# Plot distributions of variables

hist(df[!, :sales])
hist(df_ic[!, :IPN31152N]; normalization = :pdf, bins = 10, color = :firebrick, axis = 
    (; xlabel = "Industrial Production of Ice Cream and Frozen Desert (% of 2017)", ylabel = "Probability Density",
    title = "Distribution of Ice Cream Sales"))
hist(df_dm[!, :Drowning]; normalization = :pdf, bins = 10, color = :firebrick, axis = 
    (; xlabel = "Deaths by Drowning per Month, Excluding Water Transport Drownings", ylabel = "Probability Density",
    title = "Distribution of Deaths by Drowning"))

hist(log10.(df_dm[!, :Drowning]); normalization = :pdf, bins = 10, color = :firebrick, axis = 
    (; xlabel = "Deaths by Drowning per Month, Excluding Water Transport Drownings", ylabel = "Probability Density",
    title = "Distribution of Deaths by Drowning"))

fig, ax, scat = scatter(df[!, "sales"], df[!, "drowning"],
    axis=(; title = "Ice cream sales vs. Drowning (in log scale)",
        xlabel = "Industrial Production of Ice Cream and Frozen Desert (% of 2017)",
        ylabel = "Deaths by Drowning per Month, Excluding Water Transport Drownings",
        yscale = log10, yminorticksvisible = true, yminorgridvisible = false, yminorticks = IntervalsBetween(8)))
