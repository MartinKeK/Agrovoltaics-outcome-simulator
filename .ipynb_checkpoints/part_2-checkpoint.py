def PAR_plants(spectrum_df, weather_index, tot_coeff):

    # We store the spectrogram and decompose it by wavelength.

    # Rename the input of spectrums

    hourly_spectral_conv = spectrum_df["ghi_spectrum_col_conv"]
    hourly_spectral_dssc = spectrum_df["ghi_spectrum_col_dssc"]
    hourly_spectral_of = spectrum_df["open_field_spectrum"]

    # We create a dataframe to store the absorbed PAR taking into account the ABSORBED PAR AND ACTION PAR (TOTAL QUANTUM YIELD).

    PAR_hourly = pd.DataFrame()
    PAR_hourly.index = weather_index

    # Here we will take the hourly spectra that reach the crop in 3 cases: conventional cells, DSSCs, and open field. Finally, the crop absorption coefficients at each wavelength will be applied to determine how much of that spectrum the plant actually utilizes.
    PAR_absorption_conv = []
    PAR_absorption_dssc = []
    PAR_absorption_of = []
    for date_index_ab in range(len(hourly_spectral_conv)):
        PAR_absorption_conv.append(
            hourly_spectral_conv[date_index_ab] * tot_coeff.reshape(-1)
        )
        PAR_absorption_dssc.append(
            hourly_spectral_dssc[date_index_ab] * tot_coeff.reshape(-1)
        )
        PAR_absorption_of.append(
            hourly_spectral_of[date_index_ab] * tot_coeff.reshape(-1)
        )

    PAR_hourly["PAR_absorption_conv"] = PAR_absorption_conv
    PAR_hourly["PAR_absorption_dssc"] = PAR_absorption_dssc
    PAR_hourly["PAR_absorption_of"] = PAR_absorption_of

    # ----------------------------------------------------------------------------------------------

    # Ahora vamos a pasar los resultados a J/m2·day que es como funciona el simulador de cultivos.

    #     #creamos el dataframe donde guardaremos los resultados
    #     PAR_for_daily = pd.DataFrame()
    #     PAR_for_daily.index = weather_index

    #     PAR_absorption_tot = []
    #     PAR_absorption_tot_of = []

    # #     #Se va a calcular la irradiancia total, sumando lo que aportan todas las longitudes de onda y posteriormente se pasa al total diario.

    #     for date_index in range(len(hourly_spectral)):
    #         PAR_absorption_tot.append((PAR_hourly['PAR_absorption'][date_index].sum()) * 3600)
    #         PAR_absorption_tot_of.append((PAR_hourly['PAR_absorption_of'][date_index].sum()) * 3600)

    #     PAR_for_daily['PAR_absorption_tot'] = PAR_absorption_tot
    #     PAR_for_daily['PAR_absorption_tot_of'] = PAR_absorption_tot_of
    #     PAR_daily = PAR_for_daily.resample('D').sum()
    #     PAR_equivalent = PAR_daily * 2 #Para contrarestar el 0.5 del simulador de cultivos

    #     -----------------------------------------CAMBIOS ----------------------------------------------

    # Now we will convert the results to J/m2·day which is how the crop simulator operates.
    PAR_for_daily = pd.DataFrame()
    PAR_for_daily.index = weather_index

    PAR_absorption_tot_conv = []
    PAR_absorption_tot_dssc = []
    PAR_absorption_tot_of = []

    # The total irradiance will be calculated by summing up the contribution of all wavelengths and then converting it to the daily total.

    for date_index in range(len(hourly_spectral_conv)):
        PAR_absorption_tot_conv.append(
            (PAR_hourly["PAR_absorption_conv"][date_index].sum()) * 3600
        )
        PAR_absorption_tot_dssc.append(
            (PAR_hourly["PAR_absorption_dssc"][date_index].sum()) * 3600
        )
        PAR_absorption_tot_of.append(
            (PAR_hourly["PAR_absorption_of"][date_index].sum()) * 3600
        )

    PAR_for_daily["PAR_absorption_tot_conv"] = PAR_absorption_tot_conv
    PAR_for_daily["PAR_absorption_tot_dssc"] = PAR_absorption_tot_dssc
    PAR_for_daily["PAR_absorption_tot_of"] = PAR_absorption_tot_of
    PAR_daily = PAR_for_daily.resample("D").sum()
    PAR_equivalent = (
        PAR_daily * 2
    )  # The crop simulator directly estimates a PAR of 1/2 with respect to the total irradiance, without taking into account the shape of the spectrum (since it assumes it always follows the normal distribution). Since in this case we have taken into account the plant's absorption for each wavelength, we are going to nullify the 1/2 factor that the built-in crop simulator has by multiplying our results by 2.


# -----------------------------------------------------------------------------------------------------
