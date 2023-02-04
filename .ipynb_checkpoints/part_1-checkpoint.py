from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from dataclasses import dataclass

@dataclass
class CoorPV():
    Latitude: float
    Longitude: float
    Name: str
    Altitude: float
    
    
    """                                      --------------------------------------------------------------- GROUND SPECTRUM FUNCTION -------------------------------------------------------------------------------------------------------------------
"""
def ground_spectrum(indice_weather, dict_coor): 
    
    """Get hourly solar spectrum in the horizontal plane taking in account the index used in other functions and the coordinates needed to calculate the relative airmass.
      
    Parameters
    ----------
    indice_weather: pandas.core.indexes.datetimes.DatetimeIndex
        The DatetimeIndex of the selected weather data dataframe.
    dict_coor: dict
        Selected coordinates dictionary for the location.

        
    Returns
    ---------
    spectrum_normalized : pandas.DataFrame
        Time-series of hourly data of the spectrum decomposed by wavelength.
     """  
    # Location
    lat = dict_coor['Latitude']
    lon = dict_coor['Longitude']
    altitude = dict_coor['Altitude']
    
    # System geometric fixed for horizontal plane.
    tilt = 0
    azimuth = 180
    
    
    # Metheorological data
    date_time_interval = indice_weather
    solar_pos = solarposition.get_solarposition(date_time_interval, lat, lon)
    aoi = irradiance.aoi(tilt, azimuth, solar_pos.apparent_zenith, solpos.azimuth)
    
    # The technical report uses the 'kasten1966' airmass model, but later
    # versions of SPECTRL2 use 'kastenyoung1989'.  Here we use 'kasten1966'
    # for consistency with the technical report.

    # assumptions from the technical report:
    pressure = pvlib.atmosphere.alt2pres(altitude)
    water_vapor_content = 0.5  # (cm)
    tau500 = 0.1
    ozone = 0.31  # (atm-cm)
    albedo = 0.2
    
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

"""                                      --------------------------------------------------------------- DIFFUSE CORRECTOR FUNCTION -------------------------------------------------------------------------------------------------------------------
"""

def diffuse_correct(weather_df, dict_pv, inter_row_space, tilt_pv, azimuth_pv):
     """Get hourly solar diffuse corrector. On the one hand, we have the diffuse in a plane, on the other hand, we have the diffuse resulted from a module shadow. If we compare both of them hourly, we can take the percent of diffuse that the system loses, depending on the solar position and geometry of the pv system."""
    
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




"""                                      --------------------------------------------------------------- SHADING FACTORS FUNCTION -------------------------------------------------------------------------------------------------------------------
"""
def shading_factors(tilt, azimuth, space, dict_pv, coor_pv, indice):
    """Get hourly shading factors. With this function, we can get hourly the proportion of the ground that is shaded and unshaded depending on the geometry of the system and the moment of the year."""
    
    # Check if coord dict is complete
     if isinstance(coor_pv, CoorPV) == False:
            raise Error("coor_pv must be a CoorPV instance")
   
    # In this case we're going to use the clear-sky model. The objetive of this part is cheking how pv modules will shade the ground and take an hourly factor of shading.
    # We take a random year but taking in account the leap-year to have shading factor of February 29.   
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
        raise Error('It is mandatory to insert a azimuth value between -180 y 180')
        
    # Get Metheorological data from a clearsky simulation
    
    site_for_shading = location.Location(latitude, longitude, tz=timezone)
    times = pd.date_range(start_date, freq=freq, periods = periods_in_a_year,
                          tz = timezone)
    weather_clear_sky = site_for_shading.get_clearsky(times)
    weather_clear_sky.index.name = "utc_time"
    
    # Taking solar position angles.
    
    solpos_clear_sky = pvlib.solarposition.get_solarposition(
        time=weather_clear_sky.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        pressure=pvlib.atmosphere.alt2pres(altitude),
    )   
    
    

    # We need this dataframe to introduce it in PvFactors framework 
    irradiance_for_shading = pd.DataFrame({'solar_zenith': solpos_clear_sky['zenith'], 
                                           'solar_azimuth': solpos_clear_sky['azimuth'], 
                                           'surface_tilt': tilt, 'surface_azimuth': azimuth_pvfactors, 
                                           'dni': weather_clear_sky['dni'], 
                                           'dhi': weather_clear_sky['dhi'], 'albedo': albedo })
    
    #PV Parameters

    # Here we set the Ground Coverage Ratio (grc), if the spacing between rows were less than the length of the panel it would be set to 1       (maximum), else it is defined as the relationship between the length of the panel and the space (distance between one row and another).


    pv_height = dict_pv['Height']
    total_row_length = dict_pv['Length'] * dict_pv['Stacked']
    
    if inter_row_space < total_row_length:
        ground_coverage_ratio = 1.0
    else:
        ground_coverage_ratio = total_row_length / inter_row_space

    
    # PvFactors libs.
        
    from pvfactors.engine import PVEngine
    from pvfactors.geometry import OrderedPVArray
        

    # ------------------------------------------ PV SHADING FACTOR ----------------------------------------
     # Shading factor originated in the second row. It tells us the fraction of panel to which light would reach. From the case with 2 panels, we obtain the necessary data to calculate the shadow exerted by the first row on the second.
        
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
        
#         Here we calculate the shading factor between rows only taking in account the moments of the day where the sun in out.
        
        if angles<=90 and second_module_irradiance[contador_pv_shading] > first_module_irradiance[contador_pv_shading] and second_module_irradiance[contador_pv_shading] > 0:  
            pv_shading_factor.append(first_module_irradiance[contador_pv_shading]/second_module_irradiance[contador_pv_shading]) 
            contador_pv_shading = contador_pv_shading + 1
        elif angles<= 90 and first_module_irradiance[contador_pv_shading] > second_module_irradiance[contador_pv_shading] and first_module_irradiance[contador_pv_shading] > 0:
            pv_shading_factor.append(second_module_irradiance[contador_pv_shading]/first_module_irradiance[contador_pv_shading])   
            contador_pv_shading = contador_pv_shading + 1
            
        else:
            pv_shading_factor.append(0)
            contador_pv_shading = contador_pv_shading + 1
        
    
    # ------------------------------------------ GROUND SHADING FACTOR------------------------------------  
    
    # Now the percent of the ground shaded will be calculated. We are going to difference 2 parts of the ground. Shaded part and illuminated parts. This factor will say hourly the percent of each part.
    # We are going to model in PvFactors a case with 30 rows.
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
    
    #We take here the cut point of the saded zones.
    # shadow_points = pvarray.ts_ground.shadow_coords_left_of_cut_point(2)
    def fn_report(pvarray): return {'shaded_coords': (pvarray.ts_ground.shadow_coords_left_of_cut_point(1))}
    fast_mode_pvrow_index = n_pv_rows  - 1
    engine = PVEngine(pvarray,fast_mode_pvrow_index = fast_mode_pvrow_index)
    engine.fit(irradiance_for_shading.index, irradiance_for_shading.dni, irradiance_for_shading.dhi,
               irradiance_for_shading.solar_zenith, irradiance_for_shading.solar_azimuth,
               irradiance_for_shading.surface_tilt, irradiance_for_shading.surface_azimuth,
               albedo)
    shadow_coords = engine.run_fast_mode(fn_build_report=fn_report)
    
# Here we take a vector from -100 to 100 in the coordinates of the system with 30 pvrows and we are going to check the parts that are illuminated and shaded.
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
        
        
    # We keep the data
    
    shading_data_frame = pd.DataFrame({'PV_Shading_Factor': pv_shading_factor, 'Ground_Shading_Factor': ground_shading_factor})
    shading_data_frame.index = weather_clear_sky.index
    
    # Now we adapt this dataframe to the dataframe we use outside the function.
    
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
    
#     We can calculate the diffuse correction in this part to have all the corrector together
    diffuse_K = diffuse_correct(irradiance_for_shading, dict_pv, space, tilt, azimuth_pvfactors) 
   

    
#     Here we will return the shading final factors (Ground and inter-pv-rows) and the diffuse corrector for shading parts.
    return shading_final_results, diffuse_K

"""                                      --------------------------------------------------------------- LAND YIELD FUNCTION -------------------------------------------------------------------------------------------------------------------
"""

def land_yield(tilt, azimuth, space, dict_pv, dict_coor, dict_crop, is_shading, is_spectrum, start_year, end_year,spectral_out):
    
                     
            
    """Get hourly RELLENAR ESTO CUANDO ESTEN TODOS LIMPIOS Y COMPROBAR QUE DA TODO!!!!

    Parameters
    ----------
    latitude: float
        In decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude: float
        In decimal degrees, between -180 and 180, east is positive (ISO 19115)
    start: int or datetime like, default: None
        First year of the radiation time series. Defaults to first year
        available.
    end: int or datetime like, default: None
        Last year of the radia"""
    
    
    
    # Defining System
    # It will be verified that in the dictionary with the module data, we have a valid module from the Sandia library. If it is correct, that module will be selected.

    if "PV_module" in dict_pv == False:
        raise Exception("PV module is not defined in PV dict")
        
    if dict_pv["PV_module"] == "SolarWorld250W":
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        module = sandia_modules['SolarWorld_Sunmodule_250_Poly__2013_']
        
    elif dict_pv["PV_module"] == "AdventSolar160W":
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        module = sandia_modules['Advent_Solar_AS160___2006_']
        
    elif dict_pv["PV_module"] == "CanadianSolar300W":
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        module = sandia_modules['Canadian_Solar_CS6X_300M__2013_']
        
    else:
        raise Exception("Not Valid PV module")
    
    # It will be verified that in the dictionary with the inverter data, we have a valid inverter from the CEC library. If it is correct, that inverter will be selected
        
    if "PV_inverter" in dict_pv == False:
        raise Exception("PV inverter is not defined in PV dict")
        
    if dict_pv["PV_inverter"] == "Micro250W208V":
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        
    elif dict_pv["PV_inverter"] == "Micro300W240V":
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        inverter = sapm_inverters['ABB__MICRO_0_3HV_I_OUTD_US_240__240V_']
        
    elif dict_pv["PV_inverter"] == "Micro10KW480V":
        sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        inverter = sapm_inverters['ABB__PVI_10_0_I_OUTD_x_US_480_y_z__480V_']
        
    else:
        raise Exception("Not Valid PV inverter")
            
      
    # Thermal parameters for open rack glass.
    
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'] ['open_rack_glass_glass']
    
    # Location Data
    # First we check if coordinates are correct.
    
    coor_key = ['Latitude', 'Longitude', 'Name', 'Altitude']
    for key in coor_key:
        if (key in dict_coor) == False:
            print("No esta la clave " + key)
            
    #Keep location data
    
    latitude = dict_coor['Latitude']
    longitude = dict_coor['Longitude']
    name = dict_coor['Name']
    altitude = dict_coor['Altitude']
    timezone = 'UTC'
    
#     The normal azimuth convention is 0 for south orientation. However, for pvlib south orientation is 180. Here, azimuth is adapted for pvlib.
    if azimuth == 180:
        azimuth_pvlib = 0
    else:
        azimuth_pvlib = azimuth + 180
        
    
    
    # Obtain weather data from PVGIS database.
    
    weather = pvlib.iotools.get_pvgis_hourly(latitude, longitude, start = start_year, end = end_year,
                                  url='https://re.jrc.ec.europa.eu/api/v5_2/', surface_tilt = tilt, surface_azimuth = azimuth  , map_variables = True) [0]
    
    for counter_correction in range(len(weather.index)):
        if weather.index.year[counter_correction] != start_year or weather.index.year[counter_correction] != end_year:
            weather.drop(weather.index[counter_correction])
    weather.index.name = "utc_time"
    weather.index = weather.index.floor(freq='H') 
    
    # Irradiance Calculations
    # Solarposition from location data.
    solpos = pvlib.solarposition.get_solarposition(
        time=weather.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=weather["temp_air"],
        pressure=pvlib.atmosphere.alt2pres(altitude),
    )
    # Some interesting atmospheric properties are computed to enhance the calculation of the exact irradiance reaching the panel.
    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)    # Extra-atmospheric irradiance.
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])    # Air mass passing by each ray.
    pressure = pvlib.atmosphere.alt2pres(altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    aoi = pvlib.irradiance.aoi(                                     
            tilt,
            azimuth_pvlib,
            solpos["apparent_zenith"],
            solpos["azimuth"],
        ) 
    
    #Keep radiance data in the dataframe call total_irradiance. Here we have all the components of the irradiance in the plane-of-array (tilted module).
    
    poa_global = weather['poa_direct'] +  weather['poa_ground_diffuse'] + weather['poa_sky_diffuse']
    poa_diffuse = weather['poa_ground_diffuse'] + weather['poa_sky_diffuse']
    total_irradiance = pd.DataFrame({'poa_global': poa_global, 'poa_direct': weather.poa_direct, 'poa_diffuse': poa_diffuse, 'poa_sky_diffuse': weather.poa_sky_diffuse, 'poa_ground_diffuse': weather.poa_ground_diffuse})
    total_irradiance.index = weather.index
    
#     Here we use an inverse transposition model to get ghi,dni and dhi from poa irradiances.
    horizontal_irradiance = pvlib.irradiance.gti_dirint(total_irradiance['poa_global'], aoi, solpos["apparent_zenith"], solpos["azimuth"], weather.index,
               tilt, azimuth_pvlib, pressure,
               use_delta_kt_prime=True, temp_dew=None, albedo=.2,
               model='perez', model_perez='allsitescomposite1990',
               calculate_gt_90=True, max_iterations=30)
    
   

    # Hourly cell temperature
    
    cell_temperature = pvlib.temperature.sapm_cell(
            total_irradiance['poa_global'],
            weather["temp_air"],
            weather["wind_speed"],
            **temperature_model_parameters,
        )
   
    # Here, shading losses between rows are applied and shading factors are calculated between two rows and between the panel and the ground.
    if is_shading:
        (shading_factors_output, diffuse_corrector) = shading_factors(tilt, azimuth, space, dict_pv, dict_coor, total_irradiance.index)
        total_irradiance['poa_direct'] = total_irradiance['poa_direct'] * (1 - shading_factors_output['PV_Shading_Factor'])

#        Effective irradiance taking in account optical losses.
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            total_irradiance['poa_direct'],
            total_irradiance['poa_diffuse'],
            am_abs,
            aoi,
            module,
        )
        
    # Calculate dc and ac power generated for one module.
    
    info_pv_dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
    info_pv_ac = pvlib.inverter.sandia(info_pv_dc['v_mp'], info_pv_dc['p_mp'], inverter)
        
    
    #Based on the geometry of the system, a calculation will be performed to determine the number of panels that can be installed and, as a result, the total power they will generate collectively.
    
    pv_key = ['Length', 'Width', 'Stacked']
    for key in pv_key:
        if (key in dict_pv) == False:
            print("No esta la clave " + key)
       
    w_pv = dict_pv  ['Width']
    l_pv = dict_pv  ['Length']
      
    l_crop = dict_crop ['Length']
    w_crop = dict_crop ['Width']
    tilt_rad = math.radians(tilt)
    distance_condition = space
    if distance_condition < 1.25:                  #A minimum distance of 1.25 meters between rows is always set due to design conditions.
        space_final = 1.25
    else: 
        space_final = space
    
    correction_per_row = 4 # 4 meters
    pv_per_row = ((w_crop-correction_per_row)/w_pv) * dict_pv['Stacked']
    n_rows = int(l_crop/(space_final)) + 1
    n_pv =  n_rows * pv_per_row
    MW2W = 1000000             #Conversion from MW (megawatts) to W (watts)
    
    if start_year == end_year:
        n_years = 1 
    else:
        n_years = (end_year - start_year) + 1
        
    # P_hourly = info_pv_dc.p_mp * n_pv * dict_pv['performance_factor']
    
    # ---------------------------------------ADAPTATION TO DSSC POWER-------------------------------
    
    P_hourly_conv = info_pv_dc.p_mp * n_pv * 1
    P_hourly_dssc = info_pv_dc.p_mp * n_pv * 0.6
    P_dc_conv = P_hourly_conv.sum() / (MW2W*n_years) 
    P_dc_dssc = P_hourly_dssc.sum() / (MW2W*n_years) 

    
    
    
    
                        #--------------------------------- --------Total Power Output Calculation for Solar Panel Array-----------------------------------------
  
    results = pd.DataFrame({'Potencia': P_hourly_conv, 'real_ghi': horizontal_irradiance['ghi']})
    results.index = total_irradiance.index
    print('Potencia Conv (MWh): ' ,P_dc_conv)
    print('Potencia Dssc (MWh): ' ,P_dc_dssc)
    print('Número de paneles: ', n_pv)
    print('Número de hileras: ', n_rows)
    print('Densidad de paneles (%):', (l_pv/space)*100)
    print('Hectáreas:', (l_crop * w_crop)/10000)
    
#     --------------- "Irradiance and Spectrum Reaching the Ground for Conventional Cells" ---------------------
    zenith_rad = (solpos.zenith)*(math.pi/180)
    cos_zenith = np.cos(zenith_rad)
    # Total Diffuse irradiance
    dhi_col = (horizontal_irradiance.dhi)
    # Total Direct Irradiance
    dni_col= (horizontal_irradiance.dni)
    
    # ---------------------------------------DIFFUSSE CORRECTOR-------------------------------------------------
    df_keeper=pd.DataFrame()
    df_keeper.index = total_irradiance.index
    A = np.array([[]]*len(total_irradiance) + [[1]])[:-1]
    df_keeper['corrector'] = A
    for index, row in diffuse_corrector.iterrows():
        current_selected_index = (df_keeper.index.month == index.month) & (df_keeper.index.hour == index.hour)
        df_keeper[current_selected_index] = row.diffuse_corrector
    
    
  
    # ----------------------------------------------------------------------------------
    #Direct Irradiance decomposition
    dni_shaded_col = dni_col * shading_factors_output.Ground_Shading_Factor
    dni_clear_col = dni_col * (1 - shading_factors_output.Ground_Shading_Factor)
    dhi_shaded_col_placa = (dhi_col * shading_factors_output.Ground_Shading_Factor) * (1-df_keeper.corrector)
    dhi_shaded_col = (dhi_col * shading_factors_output.Ground_Shading_Factor) * df_keeper.corrector
    dhi_clear_col = dhi_col * (1 - shading_factors_output.Ground_Shading_Factor)
    
    if is_spectrum:    
        A = np.array([[]]*len(results) + [[1]])[:-1]
        # results['clear_spectrum'] = A 
        # results['shaded_spectrum'] = A 
        # results['ghi_spectrum_col'] = A 
        # results['dhi_spectrum_col'] = A
        # results['dni_spectrum_col'] = A
        # results['open_field_spectrum'] = A
        
#         ----------------------------------CHANGES -----------------------------------
        results['ghi_spectrum_col_conv'] = A 
        results['dhi_spectrum_col_conv'] = A
        results['dni_spectrum_col_conv'] = A
        
        results['ghi_spectrum_col_dssc'] = A 
        results['dhi_spectrum_col_dssc'] = A
        results['dni_spectrum_col_dssc'] = A
        
        results['open_field_spectrum'] = A



# ---------------------------------------------------------------------------------------

#         # Primero rellenamos todas las listas con un array predeterminado formado por 1.
#         datetime_col = total_irradiance.index[0]
#         for index,current_time in enumerate(results.index):
#             results['clear_spectrum'][index] = ground_spectrum(current_time, dict_coor).reshape(-1)
#             results['shaded_spectrum'][index] = np.array(results['clear_spectrum'][index]) * np.array(dict_pv['trans_spectrum'])
     
#         results['dni_spectrum_col'] = results['clear_spectrum'] * dni_clear_col + results['shaded_spectrum'] * dni_shaded_col
#         results['dhi_spectrum_col'] = results['clear_spectrum'] * dhi_col
#         results['ghi_spectrum_col'] = results['dni_spectrum_col'] * cos_zenith + results['dhi_spectrum_col'];
        
#         results['open_field_spectrum'] = results['clear_spectrum'] * horizontal_irradiance['ghi']
#         results.fillna(0)     #SOLVE NAN PROBLEM.
        
        
        # -----------------------------------CHANGES -----------------------------
"""        
In this section, the spectral irradiance reaching the ground will be calculated under two conditions:

Conventional Modules (opaque)
Dye Sensitized Solar Cells (semitransparent)
For conventional modules, there will be areas illuminated by the sun that filters through the gaps where the irradiance will be the normal horizontal spectrum ("clear spectrum"), and on the other hand, there will be areas shaded by the module where the resulting spectrum will be zero for the DNI and corrected for the DHI (taking into account the losses in diffuse light due to shading).

For DSSCs, there will be illuminated areas that will work the same as the previous case. And in the case of shaded areas, being semitransparent, the DNI will not be zero but instead, the cell's transmittance must be applied to the spectrum to calculate the resulting spectrum that passes through it and reaches the ground.

"""

        results['dni_spectrum_col_conv'] = spectral_out['clear_spectrum'] * dni_clear_col + spectral_out['shaded_spectrum_conv'] * dni_shaded_col
        results['dhi_spectrum_col_conv'] = spectral_out['clear_spectrum'] * dhi_clear_col + spectral_out['clear_spectrum'] * dhi_shaded_col + spectral_out['shaded_spectrum_conv'] * dhi_shaded_col_placa
        results['ghi_spectrum_col_conv'] = results['dni_spectrum_col_conv'] * cos_zenith + results['dhi_spectrum_col_conv']
        
        results['dni_spectrum_col_dssc'] = spectral_out['clear_spectrum'] * dni_clear_col + spectral_out['shaded_spectrum_dssc'] * dni_shaded_col
        results['dhi_spectrum_col_dssc'] = spectral_out['clear_spectrum'] * dhi_clear_col + spectral_out['clear_spectrum'] * dhi_shaded_col + spectral_out['shaded_spectrum_dssc'] * dhi_shaded_col_placa
        # results['dhi_spectrum_col_dssc'] = spectral_out['clear_spectrum'] * dhi_col
        results['ghi_spectrum_col_dssc'] = results['dni_spectrum_col_dssc'] * cos_zenith + results['dhi_spectrum_col_dssc']
        
        results['open_field_spectrum'] = spectral_out['clear_spectrum'] * horizontal_irradiance['ghi']

    #Outputs
    return P_dc_conv, P_dc_dssc, results
    # return results 