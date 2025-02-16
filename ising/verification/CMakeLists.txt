add_executable(Ising2D_FF FF_internalEnergy_specificHeat_mpfr.c)
target_link_libraries (Ising2D_FF LINK_PUBLIC "m;mpfr")
