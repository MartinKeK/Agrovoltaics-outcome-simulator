from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from dataclasses import dataclass

@dataclass
class CoorPV():
    Latitude: float
    Longitude: float
    Name: str
    Altitude: float
    
def ground_spectrum(indice_weather, dict_coor):
               
    # Location
    lat = dict_coor['Latitude']
    lon = dict_coor['Longitude']
    altitude = dict_coor['Altitude']
    
    # System geometric
    tilt = 0
    azimuth = 180
    
    # assumptions from the technical report:
    pressure = pvlib.atmosphere.alt2pres(altitude)
    water_vapor_content = 0.5  # (cm)
    tau500 = 0.1
    ozone = 0.31  # (atm-cm)
    albedo = 0.2
    
    # Metheorological data
    date_time_interval = indice_weather
    solar_pos = solarposition.get_solarposition(date_time_interval, lat, lon)
    aoi = irradiance.aoi(tilt, azimuth, solar_pos.apparent_zenith, solpos.azimuth)
    
    # The technical report uses the 'kasten1966' airmass model, but later
    # versions of SPECTRL2 use 'kastenyoung1989'.  Here we use 'kasten1966'
    # for consistency with the technical report.
    relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith,
                                                   model='kasten1966')
    # Get solar light spectrum 
    spectra = spectrum.spectrl2(
        apparent_zenith=solpos.apparent_zenith,
        aoi=aoi,
        surface_tilt=tilt,
        ground_albedo=albedo,
        surface_pressure=pressure,
        relative_airmass=relative_airmass,
        precipitable_water=water_vapor_content,
        ozone=ozone,
        aerosol_turbidity_500nm=tau500,)
    
    # Spectrum Normalized 
    
    spectrum_normalized = spectra['poa_global'] / (spectra['poa_global'].sum())
    
    return spectrum_normalized


def diffuse_correct(weather_df, dict_pv, inter_row_space, tilt_pv, azimuth_pv):
    
    pv_height = dict_pv['Height']
    total_row_length = dict_pv['Length'] * dict_pv['Stacked']
    
    if inter_row_space < total_row_length:
        ground_coverage_ratio = 1.0
    else:
        ground_coverage_ratio = total_row_length / inter_row_space
        
    albedo = 0.2
    weather_average = weather_df.iloc[ weather_df.index.day == 15]
    pvarray_parameters = {
        'n_pvrows': 20,            # number of pv rows
        'pvrow_height': pv_height,        # height of pvrows (measured at center / torque tube)
        'pvrow_width': l,         # width of pvrows
        'axis_azimuth': 90,       # azimuth angle of rotation axis
        'surface_azimuth': azimuth_pv,
        'surface_tilt': tilt_pv,
        'gcr': ground_coverage_ratio,               # ground coverage ratio
    }



    # Create ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters)
    # Create engine
    engine = PVEngine(pvarray)
    # # Fit engine to data
    engine.fit(weather_average.index, weather_average.dni, weather_average.dhi,
               weather_average.solar_zenith, weather_average.solar_azimuth,
               weather_average.surface_tilt, weather_average.surface_azimuth,
               albedo)
    
    # Get the PV array
    pvarray.update_params
    sol = engine.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    shadowed = pvarray.ts_ground.shaded.get_param_weighted('qinc')  
    df_correction_difuse = pd.DataFrame()
    df_correction_difuse.index = weather_average.index
    df_correction_difuse['shadow'] = shadowed
    df_correction_difuse['difuse'] = weather_average['dhi']
    df_correction_difuse['diffuse_corrector'] = df_correction_difuse['shadow'] / df_correction_difuse['difuse']
    df_correction_difuse.fillna(0)
    return df_correction_difuse

def shading_factors(tilt, azimuth, space, dict_pv, coor_pv, indice):
    
    # Se comprueba que el diccionario con las coordenadas esté completo y se guardan las coordenadas.
     if isinstance(coor_pv, CoorPV) == False:
            raise Error("coor_pv must be a CoorPV instance")
   
    # Justify this values
    timezone = 'UTC'
    start_date = '01-01-2016'
    freq = "60min"
    periods_in_a_year = 8784
    albedo = 0.2
    
    # Adapt azimuth to pvfactors format
    # ToDo move to function
    if azimuth >= 0 and azimuth <= 90:
        azimuth_pvfactors = azimuth + ((90-azimuth) * 2)
    elif azimuth > 90 and azimuth <= 180:
        azimuth_pvfactors = azimuth - ((azimuth-90) * 2)
    elif azimuth<0 and azimuth >= -90:
        azimuth_pvfactors = (azimuth + ((90-azimuth) * 2)) - 360
    elif azimuth > -180 and azimuth < -90:
        azimuth_pvfactors = (azimuth - ((azimuth-90) * 2)) - 360
    else:
        raise Error('It is mandatory to insert a aximuth value between -180 y 180')
        
    # Get Metheorological data from a clearsky simulation
    site_for_shading = location.Location(latitude, longitude, tz=timezone)
    times = pd.date_range(start_date, freq=freq, periods = periods_in_a_year,
                          tz = timezone)
    weather_clear_sky = site_for_shading.get_clearsky(times)
    weather_clear_sky.index.name = "utc_time"
    
    # Se obtiene la posición solar.
    
    solpos_clear_sky = pvlib.solarposition.get_solarposition(
        time=weather_clear_sky.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        pressure=pvlib.atmosphere.alt2pres(altitude),
    )   
    
    
    # DataFrame necesario para introducir en el pvfactors.
    irradiance_for_shading = pd.DataFrame({'solar_zenith': solpos_clear_sky['zenith'], 
                                           'solar_azimuth': solpos_clear_sky['azimuth'], 
                                           'surface_tilt': tilt, 'surface_azimuth': azimuth_pvfactors, 
                                           'dni': weather_clear_sky['dni'], 
                                           'dhi': weather_clear_sky['dhi'], 'albedo': albedo })
    
    #PV Parameters
    # This has to be writen in english 
#    Aquí fijamos el Ground Coverage Ratio (grc), si la separación entre hileras fuese menor que
#     la longitud del panel se fijaría en 1, si no se define como la relación entre la longitud del panel y 
#     el espacio (distancia entre una hilera y otra). 
    pv_height = dict_pv['Height']
    total_row_length = dict_pv['Length'] * dict_pv['Stacked']
    
    if inter_row_space < total_row_length:
        ground_coverage_ratio = 1.0
    else:
        ground_coverage_ratio = total_row_length / inter_row_space

    
    # Importamos las librerías necesarias para usar PvFactors.
        
    from pvfactors.engine import PVEngine
    from pvfactors.geometry import OrderedPVArray
        

    # ------------------------------------------ FACTOR SOMBREADO PV ----------------------------------------
     # Factor de sombreamiento originado en la segunda hilera. Nos dice la fracción de panel a la que le llegaría luz.
     # Del caso con 2 paneles obtenemos datos necesarios para calcular la sombra que ejerce la primera hilera sobre la segunda.
        
    pvarray_parameters = {
    'n_pvrows': 2,          # number of pv rows
    'pvrow_height': pv_height ,# height of pvrows (measured at center / torque tube)
    'axis_azimuth': 90,   
    'surface_azimuth': azimuth_pvfactors,
    'surface_tilt': tilt,  
    'pvrow_width':l ,       # width of pvrows
    'gcr': gcr,             # ground coverage ratio
}

    # Create ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters)
    # Create engine
    engine = PVEngine(pvarray)
    # Fit engine to data
    engine.fit(irradiance_for_shading.index, irradiance_for_shading.dni, irradiance_for_shading.dhi,
               irradiance_for_shading.solar_zenith, irradiance_for_shading.solar_azimuth,
               irradiance_for_shading.surface_tilt, irradiance_for_shading.surface_azimuth,
               albedo)
               
     # Get the PV array
    pvarray = engine.run_full_mode(fn_build_report=lambda pvarray: pvarray)
        
    # Get the calculated outputs from the pv array
        
    first_module_irradiance = pvarray.ts_pvrows[1].front.get_param_weighted('qinc')
    second_module_irradiance = pvarray.ts_pvrows[0].front.get_param_weighted('qinc')
    
    # ------------------------------------------CAMBIOS ----------------------------------------
    # real_diffuse = pvarray.ts_ground.shaded.get_param_weighted('qinc')
    # to_diffuse_correction = pd.DataFrame()
    # to_diffuse_correction.index = irradiance_for_shading.index
    # to_diffuse_correction['real_diffuse'] = real_diffuse
    # diffuse_corrector = to_diffuse_correction['real_diffuse'] / irradiance_for_shading['dhi']
    # to_diffuse_correction['diffuse_corrector'] = diffuse_corrector
    
# -------------------------------------------------------------------------------------------------------    
    contador_pv_shading = 0
    pv_shading_factor = []
    for angles in solpos_clear_sky.zenith:
        
        if angles<=90 and second_module_irradiance[contador_pv_shading] > first_module_irradiance[contador_pv_shading] and second_module_irradiance[contador_pv_shading] > 0:  
            pv_shading_factor.append(first_module_irradiance[contador_pv_shading]/second_module_irradiance[contador_pv_shading]) 
            contador_pv_shading = contador_pv_shading + 1
        elif angles<= 90 and first_module_irradiance[contador_pv_shading] > second_module_irradiance[contador_pv_shading] and first_module_irradiance[contador_pv_shading] > 0:
            pv_shading_factor.append(second_module_irradiance[contador_pv_shading]/first_module_irradiance[contador_pv_shading])   
            contador_pv_shading = contador_pv_shading + 1
            
        else:
            pv_shading_factor.append(0)
            contador_pv_shading = contador_pv_shading + 1
        
    
    # ------------------------------------------ FACTOR SOMBREADO SUELO------------------------------------     
    # Calculamos ahora el factor de sombreamiento del suelo.
    # Del caso con 4 paneles obtenemos los datos necesarios para calcular el factor de sombreado del suelo.
    n_pv_rows = 30
    pvarray_parameters = {
    'n_pvrows': n_pv_rows,          # number of pv rows
    'pvrow_height': pv_height ,# height of pvrows (measured at center / torque tube)
    'axis_azimuth': 90,   
    'surface_azimuth': azimuth_pvfactors,
    'surface_tilt': tilt,  
    'pvrow_width':l ,       # width of pvrows
    'gcr': gcr,             # ground coverage ratio
}

    # Create ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters)
#     Create report function

    # Create engine
    # engine = PVEngine(pvarray)
    # Fit engine to data
    # engine.fit(irradiance_for_shading.index, irradiance_for_shading.dni, irradiance_for_shading.dhi,
    #            irradiance_for_shading.solar_zenith, irradiance_for_shading.solar_azimuth,
    #            irradiance_for_shading.surface_tilt, irradiance_for_shading.surface_azimuth,
    #            albedo)
   
     # Get the PV array
    #pvarray = engine.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    
    #De aquí tomamos los puntos de corte de las sombras generadas por los 4 paneles
    # shadow_points = pvarray.ts_ground.shadow_coords_left_of_cut_point(2)
    def fn_report(pvarray): return {'shaded_coords': (pvarray.ts_ground.shadow_coords_left_of_cut_point(1))}
    fast_mode_pvrow_index = n_pv_rows  - 1
    engine = PVEngine(pvarray,fast_mode_pvrow_index = fast_mode_pvrow_index)
    engine.fit(irradiance_for_shading.index, irradiance_for_shading.dni, irradiance_for_shading.dhi,
               irradiance_for_shading.solar_zenith, irradiance_for_shading.solar_azimuth,
               irradiance_for_shading.surface_tilt, irradiance_for_shading.surface_azimuth,
               albedo)
    shadow_coords = engine.run_fast_mode(fn_build_report=fn_report)

    vec_meters = numpy.linspace(-100,100,200000)
    tilt_rad = math.radians(tilt)
    lower_range = (l*math.cos(tilt_rad)) / 2
    upper_range = space 
    ios = [lower_range, upper_range]
    ground_shading_factor = []
    for counter in range(len(weather_clear_sky)):
        interest_meters = ((vec_meters > ios[0]) & (vec_meters < ios[1]))
        sombras = np.ones(len(vec_meters))
        for index_shadows in range(n_pv_rows):
            current_shadow = shadow_coords['shaded_coords'][index_shadows].at(counter)
            selected_meters = ((vec_meters > current_shadow[0][0]) & (vec_meters < current_shadow[1][0]))
            sombras[selected_meters] = 0
        ground_shading_factor.append((sombras[interest_meters].sum())/(interest_meters.sum()))
        
        
    # Guardamos los datos obtenidos en un DataFrame
        
    
    shading_data_frame = pd.DataFrame({'PV_Shading_Factor': pv_shading_factor, 'Ground_Shading_Factor': ground_shading_factor})
    shading_data_frame.index = weather_clear_sky.index
    
    # Ahora vamos a adaptar el dataframe al dataframe principal.
    
    # Primero creamos un dataframe con valores nan y del tamaño del dataframe principal.
    
    shading_final_results = pd.DataFrame()
    shading_final_results.index = indice

    pv_shading_list = list()
    ground_shading_list = list()
    # Iteramos para añadir los valores de los factores de sombreamiento de pv y suelo para cada uno de los días.
    for current_date in shading_final_results.index:
        current_date = current_date.replace(year = shading_data_frame.index[0].year)
        shading_index = shading_data_frame.index.get_loc(current_date)
        pv_shading_list.append(1 - shading_data_frame["PV_Shading_Factor"][shading_index])
        ground_shading_list.append(1 - shading_data_frame["Ground_Shading_Factor"][shading_index])
    shading_final_results["PV_Shading_Factor"] = pv_shading_list
    shading_final_results["Ground_Shading_Factor"] = ground_shading_list
    
    diffuse_K = diffuse_correct(irradiance_for_shading, dict_pv, space, tilt, azimuth_pvfactors) 
   

    
    
    return shading_final_results, diffuse_K