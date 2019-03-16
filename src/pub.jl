function gsd_bounds(fsim, h=[0.00025], ngrid=10)
    [get_bounds(fsim; type1=0.05, type2=0.2; ngrid=ngrid, h=hh) for hh in h]
end

function xyz_gsd_sim(iter, df, M; gen_data=ph_biweibull_1, model_fun=getBiPHWeibullModel)
    x = gsd_sim(iter, df, M=M, delta=(0.0, 0.0), gen_data=gen_data, model_fun=model_fun)
    y = gsd_sim(iter, df, M=M, delta=(0.0, log(0.6)), gen_data=gen_data, model_fun=model_fun)
    z = gsd_sim(iter, df, M=M, delta=(log(0.8), log(0.6)), gen_data=gen_data, model_fun=model_fun)
    x, y, z
end

function full_aed_sim(iter, bounds_df; h=[(0.00025, true), (0.000250001, false)], objs=[AED.public_utility, AED.sponsor_utility])
    bnds = [[r.a1 r.b1; r.a2 r.b2] for r in eachrow(bounds_df)]
    [xyz_aed_sim(iter, ab[1], h=h, objective=ab[2]) for ab in zip(bnds, objs)]
end

function xzy_aed_sim(iter, bounds; h=[(0.00025, true), (0.00025, false)], objective=AED.public_utility)
    x = AED.weibull_aed_sim(iter, bounds, delta=(0.0, 0.0), h=h, objective=objective)
    y = AED.weibull_aed_sim(iter, bounds, delta=(0.0, log(0.6)), h=h, objective=objective)
    z = AED.weibull_aed_sim(iter, bounds, delta=(log(0.8), log(0.6)), h=h, objective=objective)
    x,y,z
end

## forward simulations for Weibull proportional hazards model
weibull_fsim = AED.get_fsim(20000, AED.getBiPHWeibullModel, [[100, 50], [200, 100]]; max_n=400, M=10000,
                        delta0=[log(1.0), log(0.6)], delta=(log(0.8), log(0.6)), tau=0.0,
                        gen_data=AED.ph_weibull_1, warmup=5000, type1=0.05, type2=0.2)

## forward simulations with Spline proportional hazards model
spline_fsim = AED.get_fsim(20000, AED.getBiPHSplineModel, [[100, 50], [200, 100]]; max_n=400, M=10000,
                        delta0=[log(1.0), log(0.6)], delta=(log(0.8), log(0.6)), tau=0.0,
                        gen_data=AED.ph_billog_1, warmup=5000, type1=0.05, type2=0.2)

weibull_bounds = gsd_bounds(weibull_fsim)
spline_bounds = gsd_bounds(spline_fsim)

weibull_gsd_sim = xyz_gsd_sim(10000, weibull_bounds, 10000; gen_data=AED.ph_biweibull_1, model_fun=AED.getBiPHWeibullModel)
spline_gsd_sim = xyz_gsd_sim(10000, spline_bounds, 10000; gen_data=AED.ph_billog_1, model_fun=AED.getBiPHSplineModel)

weibull_aed_sim = full_aed_sim(1000, weibull_bounds)
