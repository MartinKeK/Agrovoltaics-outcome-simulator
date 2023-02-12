import pvfactors
from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from dataclasses import dataclass
from pvlib import (
    spectrum,
    solarposition,
    irradiance,
    atmosphere,
    location,
    temperature,
    pvsystem,
    iotools,
)
import math
import pandas as pd
import numpy as np


@dataclass
class Coor_PV:
    Latitude: float
    Longitude: float
    Name: str
    Altitude: float
@dataclass
class Ground_Dimensions():
    Length: float
    Width: float

    """                                      --------------------------------------------------------------- GROUND SPECTRUM FUNCTION -------------------------------------------------------------------------------------------------------------------
"""


def ground_spectrum(weather_index, coor_pv):
    """Get hourly solar spectrum in the horizontal plane taking in account the index used in other functions and the coordinates needed to calculate the relative airmass.

    Parameters
    ----------
    indice_weather: pandas.core.indexes.datetimes.DatetimeIndex
        The DatetimeIndex of the selected weather data dataframe.
    dict_coor: class
        Selected coordinates class for the location.


    Returns
    ---------
    spectrum_normalized : pandas.DataFrame
        Time-series of hourly data of the spectrum decomposed by wavelength.
    """
   
    # System geometric fixed for horizontal plane.
    tilt = 0
    azimuth = 180

    # Metheorological data
    date_time_interval = weather_index
    solar_pos = solarposition.get_solarposition(date_time_interval, coor_pv.Latitude, coor_pv.Longitude)
    aoi = irradiance.aoi(tilt, azimuth, solar_pos.apparent_zenith, solar_pos.azimuth)

    # The technical report uses the 'kasten1966' airmass model, but later
    # versions of SPECTRL2 use 'kastenyoung1989'.  Here we use 'kasten1966'
    # for consistency with the technical report.

    # assumptions from the technical report:
    pressure = atmosphere.alt2pres(coor_pv.Altitude)
    water_vapor_content = 0.5  # (cm)
    tau500 = 0.1
    ozone = 0.31  # (atm-cm)
    albedo = 0.2

    relative_airmass = atmosphere.get_relative_airmass(
        solar_pos.apparent_zenith, model="kasten1966"
    )
    # Get solar light spectrum
    spectra = spectrum.spectrl2(
        apparent_zenith=solar_pos.apparent_zenith,
        aoi=aoi,
        surface_tilt=tilt,
        ground_albedo=albedo,
        surface_pressure=pressure,
        relative_airmass=relative_airmass,
        precipitable_water=water_vapor_content,
        ozone=ozone,
        aerosol_turbidity_500nm=tau500,
    )

    # Spectrum Normalized

    spectrum_normalized = spectra["poa_global"] / (spectra["poa_global"].sum())

    return spectrum_normalized


"""                                      --------------------------------------
------------------------- DIFFUSE CORRECTOR FUNCTION ---------------------------
----------------------------------------------------------------------------------------
"""



def diffuse_correct(weather_df, dict_pv, inter_row_space, tilt_pv, azimuth_pv):
    """Get hourly solar diffuse corrector. On the one hand, we have the diffuse in a plane, on the
    other hand, we have the diffuse resulted from a module shadow. If we compare both of them hourly, we
    can take the percent of diffuse that the system loses, depending on the solar position and geometry of
    the pv system.
    
    Parameters
    ----------
    weather_df: pandas.DataFrame
        Selected weather data dataframe.
    dict_pv: dict
        PV geometry dictionary.
    inter_row_space: int/float.
        Space between 2 pv rows.
    tilt_pv
    azimuth_pv


    Returns
    ---------
    df_correction_difuse : pandas.DataFrame
        Time-series of hourly data of the diffuse correction factor.
    """
    #The diffuse corrector will vary linearly depending on the day of the year, going 
    #from a minimum in the winter months to a maximum reached in the summer months. 
    #That's why the correction selected will be for the mid-day of each month.
    
    day_in_middle = 15
    pv_height = dict_pv["Height"]
    total_row_length = dict_pv["Length"] * dict_pv["Stacked"]

    if inter_row_space < total_row_length:
        ground_coverage_ratio = 1.0
    else:
        ground_coverage_ratio = total_row_length / inter_row_space

    albedo = 0.2
    weather_average = weather_df.iloc[weather_df.index.day == day_in_middle]
    pvarray_parameters = {
        "n_pvrows": 20,  # number of pv rows
        "pvrow_height": pv_height,  # height of pvrows (measured at center / torque tube)
        "pvrow_width": dict_pv["Length"],  # width of pvrows
        "axis_azimuth": 90,  # azimuth angle of rotation axis
        "surface_azimuth": azimuth_pv,
        "surface_tilt": tilt_pv,
        "gcr": ground_coverage_ratio,  # ground coverage ratio
    }

    # Create ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters)
    # Create PVlib engine
    engine = PVEngine(pvarray)
    # # Fit engine  data
    engine.fit(
        weather_average.index,
        weather_average.dni,
        weather_average.dhi,
        weather_average.solar_zenith,
        weather_average.solar_azimuth,
        weather_average.surface_tilt,
        weather_average.surface_azimuth,
        albedo,
    )

    # Get the PV array
    pvarray.update_params
    sol = engine.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    shadowed = pvarray.ts_ground.shaded.get_param_weighted("qinc")
    df_correction_difuse = pd.DataFrame()
    df_correction_difuse.index = weather_average.index
    df_correction_difuse["shadow"] = shadowed
    df_correction_difuse["difuse"] = weather_average["dhi"]
    df_correction_difuse["diffuse_corrector"] = (
        df_correction_difuse["shadow"] / df_correction_difuse["difuse"]
    )
    df_correction_difuse.fillna(0)
    return df_correction_difuse


"""----------------------------------------------------
--------- SHADING FACTORS FUNCTION ------------------------------------
-------------------------------------------------------------------------------
"""


def shading_factors(tilt, azimuth, inter_row_space, dict_pv, coor_pv, index):
    """Get hourly shading factors. With this function, we can get hourly the proportion of the ground that is shaded and unshaded depending on the geometry of the system and the moment of the year.
    
    Parameters
    ----------
    tilt_pv
    azimuth_pv
    inter_row_space: int/float.
        Space between 2 pv rows.
    dict_pv: dict
        PV geometry dictionary.
    dict_coor: class
        Selected coordinates class for the location.
    indice_weather: pandas.core.indexes.datetimes.DatetimeIndex
        The DatetimeIndex of the selected weather data dataframe.

    Returns
    ---------
    shading_final_results : pandas.DataFrame
        Time-series of hourly data of the shading factor.
    
    """

    # Check if coord dict is complete
    # if isinstance(coor_pv, Coor_PV) == False:
    #     raise ArithmeticError("coor_pv must be a CoorPV instance")

    # In this case we're going to use the clear-sky model. The objetive of this part is cheking 
    #how pv modules will shade the ground and take an hourly factor of shading.
    # We take a random year but taking in account the leap-year to have shading factor of February 29.
    
    timezone = "UTC"
    start_date = "01-01-2016"
    freq = "60min"
    periods_in_a_year = 8784
    albedo = 0.2

    # Adapt azimuth to pvfactors format using the function azimuth_conv.
   
    azimuth_pvfactors = azimuth_conv(azimuth)

    # Get Metheorological data from a clearsky simulation

    site_for_shading = location.Location(
        coor_pv.Latitude, coor_pv.Longitude, tz=timezone
    )
    times = pd.date_range(start_date, freq=freq, periods=periods_in_a_year, tz=timezone)
    weather_clear_sky = site_for_shading.get_clearsky(times)
    weather_clear_sky.index.name = "utc_time"

    # Taking solar position angles.

    solpos_clear_sky = solarposition.get_solarposition(
        time=weather_clear_sky.index,
        latitude=coor_pv.Latitude,
        longitude=coor_pv.Longitude,
        altitude=coor_pv.Altitude,
        pressure=atmosphere.alt2pres(coor_pv.Altitude),
    )

    # We need this dataframe to introduce it in PvFactors framework
    irradiance_for_shading = pd.DataFrame(
        {
            "solar_zenith": solpos_clear_sky["zenith"],
            "solar_azimuth": solpos_clear_sky["azimuth"],
            "surface_tilt": tilt,
            "surface_azimuth": azimuth_pvfactors,
            "dni": weather_clear_sky["dni"],
            "dhi": weather_clear_sky["dhi"],
            "albedo": albedo,
        }
    )

    # PV Parameters

    # Here we set the Ground Coverage Ratio (grc), if the spacing between rows were less than the length of the panel it would be set to 1       (maximum), else it is defined as the relationship between the length of the panel and the space (distance between one row and another).

    pv_height = dict_pv["Height"]
    total_row_length = dict_pv["Length"] * dict_pv["Stacked"]

    if inter_row_space < total_row_length:
        ground_coverage_ratio = 1.0
    else:
        ground_coverage_ratio = total_row_length / inter_row_space

    # ------------------------------------------ PV SHADING FACTOR ----------------------------------------
    # Shading factor originated in the second row. It tells us the fraction of panel to which light would reach. From the case with 2 panels, we obtain the necessary data to calculate the shadow exerted by the first row on the second.

    pvarray_parameters = {
        "n_pvrows": 2,  # number of pv rows
        "pvrow_height": pv_height,  # height of pvrows (measured at center / torque tube)
        "axis_azimuth": 90,
        "surface_azimuth": azimuth_pvfactors,
        "surface_tilt": tilt,
        "pvrow_width": total_row_length,  # width of pvrows
        "gcr": ground_coverage_ratio,  # ground coverage ratio
    }

    # Create ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters)
    # Create engine
    engine = PVEngine(pvarray)
    # Fit engine to data
    engine.fit(
        irradiance_for_shading.index,
        irradiance_for_shading.dni,
        irradiance_for_shading.dhi,
        irradiance_for_shading.solar_zenith,
        irradiance_for_shading.solar_azimuth,
        irradiance_for_shading.surface_tilt,
        irradiance_for_shading.surface_azimuth,
        albedo,
    )

    # Get the PV array
    pvarray = engine.run_full_mode(fn_build_report=lambda pvarray: pvarray)

    # Get the calculated outputs from the pv array

    first_module_irradiance = pvarray.ts_pvrows[1].front.get_param_weighted("qinc")
    second_module_irradiance = pvarray.ts_pvrows[0].front.get_param_weighted("qinc")

    # ------------------------------------------CAMBIOS ----------------------------------------
    # real_diffuse = pvarray.ts_ground.shaded.get_param_weighted('qinc')
    # to_diffuse_correction = pd.DataFrame()
    # to_diffuse_correction.index = irradiance_for_shading.index
    # to_diffuse_correction['real_diffuse'] = real_diffuse
    # diffuse_corrector = to_diffuse_correction['real_diffuse'] / irradiance_for_shading['dhi']
    # to_diffuse_correction['diffuse_corrector'] = diffuse_corrector

    # -------------------------------------------------------------------------------------------------------
    count_pv_shading = 0
    pv_shading_factor = []
    for angles in solpos_clear_sky.zenith:
        #         Here we calculate the shading factor between rows only taking in account the moments of the day where the sun in out.

        if (
            angles <= 90
            and second_module_irradiance[count_pv_shading]
            > first_module_irradiance[count_pv_shading]
            and second_module_irradiance[count_pv_shading] > 0
        ):
            pv_shading_factor.append(
                first_module_irradiance[count_pv_shading]
                / second_module_irradiance[count_pv_shading]
            )
            count_pv_shading = count_pv_shading + 1
        elif (
            angles <= 90
            and first_module_irradiance[count_pv_shading]
            > second_module_irradiance[count_pv_shading]
            and first_module_irradiance[count_pv_shading] > 0
        ):
            pv_shading_factor.append(
                second_module_irradiance[count_pv_shading]
                / first_module_irradiance[count_pv_shading]
            )
            count_pv_shading = count_pv_shading + 1

        else:
            pv_shading_factor.append(0)
            count_pv_shading = count_pv_shading + 1

    # ------------------------------------------ GROUND SHADING FACTOR------------------------------------

    # Now the percent of the ground shaded will be calculated. We are going to difference 2 parts of the ground. 
    #Shaded part and illuminated parts. This factor will say hourly the percent of each part.
    # We are going to model in PvFactors a case with 30 rows.
    n_pv_rows = 30
    pvarray_parameters = {
        "n_pvrows": n_pv_rows,  # number of pv rows
        "pvrow_height": pv_height,  # height of pvrows (measured at center / torque tube)
        "axis_azimuth": 90,
        "surface_azimuth": azimuth_pvfactors,
        "surface_tilt": tilt,
        "pvrow_width": total_row_length,  # width of pvrows
        "gcr": ground_coverage_ratio,  # ground coverage ratio
    }

    # Create ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters)

    # We take here the cut point of the saded zones.
    # shadow_points = pvarray.ts_ground.shadow_coords_left_of_cut_point(2)
    def fn_report(pvarray):
        return {"shaded_coords": (pvarray.ts_ground.shadow_coords_left_of_cut_point(1))}

    fast_mode_pvrow_index = n_pv_rows - 1
    engine = PVEngine(pvarray, fast_mode_pvrow_index=fast_mode_pvrow_index)
    engine.fit(
        irradiance_for_shading.index,
        irradiance_for_shading.dni,
        irradiance_for_shading.dhi,
        irradiance_for_shading.solar_zenith,
        irradiance_for_shading.solar_azimuth,
        irradiance_for_shading.surface_tilt,
        irradiance_for_shading.surface_azimuth,
        albedo,
    )
    shadow_coords = engine.run_fast_mode(fn_build_report=fn_report)

    # Here we take a vector from -100 to 100 in the coordinates of the system with 30
    # pvrows and we are going to check the parts that are illuminated and shaded.
    vec_meters = np.linspace(-100, 100, 200000)
    tilt_rad = math.radians(tilt)
    lower_range = (total_row_length * math.cos(tilt_rad)) / 2
    upper_range = inter_row_space
    ios = [lower_range, upper_range]
    ground_shading_factor = []
    for counter in range(len(weather_clear_sky)):
        interest_meters = (vec_meters > ios[0]) & (vec_meters < ios[1])
        sombras = np.ones(len(vec_meters))
        for index_shadows in range(n_pv_rows):
            current_shadow = shadow_coords["shaded_coords"][index_shadows].at(counter)
            selected_meters = (vec_meters > current_shadow[0][0]) & (
                vec_meters < current_shadow[1][0]
            )
            sombras[selected_meters] = 0
        ground_shading_factor.append(
            (sombras[interest_meters].sum()) / (interest_meters.sum())
        )

    # We keep the data

    shading_data_frame = pd.DataFrame(
        {
            "PV_Shading_Factor": pv_shading_factor,
            "Ground_Shading_Factor": ground_shading_factor,
        }
    )
    shading_data_frame.index = weather_clear_sky.index

    # Now we adapt this dataframe to the dataframe we use outside the function.

    shading_final_results = pd.DataFrame()
    shading_final_results.index = index

    pv_shading_list = list()
    ground_shading_list = list()
    # Iteramos para añadir los valores de los factores de sombreamiento de pv y suelo para cada uno de los días.
    for current_date in shading_final_results.index:
        current_date = current_date.replace(year=shading_data_frame.index[0].year)
        shading_index = shading_data_frame.index.get_loc(current_date)
        pv_shading_list.append(
            1 - shading_data_frame["PV_Shading_Factor"][shading_index]
        )
        ground_shading_list.append(
            1 - shading_data_frame["Ground_Shading_Factor"][shading_index]
        )
    shading_final_results["PV_Shading_Factor"] = pv_shading_list
    shading_final_results["Ground_Shading_Factor"] = ground_shading_list

    #     We can calculate the diffuse correction in this part to have all the corrector together
    diffuse_K = diffuse_correct(
        irradiance_for_shading, dict_pv, inter_row_space, tilt, azimuth_pvfactors
    )

    #     Here we will return the shading final factors (Ground and inter-pv-rows) and the diffuse corrector for shading parts.
    return shading_final_results, diffuse_K


"""                                      -----------------------------------------------
---------------- LAND YIELD FUNCTION --------------------------------------------------
-----------------------------------------------------------------
"""


def land_yield(
    tilt,
    azimuth,
    inter_row_space,
    dict_pv,
    coor_pv,
    ground_dim,
    is_spectrum,
    start_year,
    end_year,
    spectral_pv_trans ,
):
    """

    Parameters
    ----------
    tilt_pv
    azimuth_pv
    inter_row_space: int/float.
        Space between 2 pv rows.
    dict_pv: dict
        PV geometry dictionary.
    dict_coor: class
        Selected coordinates for the location.
    ground_dim: class
        Dimensions of the ground.
    is_spectrum: True or False.
        True to calculate the total spectral irradiance that would reach the ground through the modules.
        False to only calculate power results.
    start_year: int
    end_year:int
    spectral_pv_trans: panda.DataFrame
        Time-series of hourly data of the transmissivity spectrum of PV modules, decomposed by wavelength.
        
    Returns
    ----------
    P_dc_conv: numpy.float64
       Total annual power generated by conventional pv modules. (MWh)
    P_dc_dssc: numpy.float64
        Total annual power generated by DSSCs modules. (MWh)
    results: panda.DataFrame
        - Time-series of hourly power data.
        - Time-series of hourly GHI.
        - Time-series of hourly resulting irradiance (spectral distributions by wavelength) 
          for open-field, dsscs and conventional pv.
        
        """

    # Defining System
    # It will be verified that in the dictionary with the module data, we have a valid module from the Sandia library. If it is correct, that module will be selected.

    if "PV_module" in dict_pv == False:
        raise Exception("PV module is not defined in PV dict")

    if dict_pv["PV_module"] == "SolarWorld250W":
        sandia_modules = pvsystem.retrieve_sam("SandiaMod")
        module = sandia_modules["SolarWorld_Sunmodule_250_Poly__2013_"]

    elif dict_pv["PV_module"] == "AdventSolar160W":
        sandia_modules = pvsystem.retrieve_sam("SandiaMod")
        module = sandia_modules["Advent_Solar_AS160___2006_"]

    elif dict_pv["PV_module"] == "CanadianSolar300W":
        sandia_modules = pvsystem.retrieve_sam("SandiaMod")
        module = sandia_modules["Canadian_Solar_CS6X_300M__2013_"]

    else:
        raise Exception("Not Valid PV module")

    # It will be verified that in the dictionary with the inverter data, we have a valid inverter from the CEC library. If it is correct, that inverter will be selected

    if "PV_inverter" in dict_pv == False:
        raise Exception("PV inverter is not defined in PV dict")

    if dict_pv["PV_inverter"] == "Micro250W208V":
        sapm_inverters = pvsystem.retrieve_sam("cecinverter")
        inverter_selected = sapm_inverters["ABB__MICRO_0_25_I_OUTD_US_208__208V_"]

    elif dict_pv["PV_inverter"] == "Micro300W240V":
        sapm_inverters = pvsystem.retrieve_sam("cecinverter")
        inverter_selected = sapm_inverters["ABB__MICRO_0_3HV_I_OUTD_US_240__240V_"]

    elif dict_pv["PV_inverter"] == "Micro10KW480V":
        sapm_inverters = pvsystem.retrieve_sam("cecinverter")
        inverter_selected = sapm_inverters["ABB__PVI_10_0_I_OUTD_x_US_480_y_z__480V_"]

    else:
        raise Exception("Not Valid PV inverter")

    # Thermal parameters for open rack glass.

    temperature_model_parameters = temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]

    # Location Data
    # First we check if coordinates are correct.

    # if isinstance(coor_pv, Coor_PV) == False:
    #     raise ArithmeticError("coor_pv must be a CoorPV instance")
    # Keep location data

    # latitude = dict_coor["Latitude"]
    # longitude = dict_coor["Longitude"]
    # name = dict_coor["Name"]
    # altitude = dict_coor["Altitude"]
    timezone = "UTC"

    #     The normal azimuth convention is 0 for south orientation. However, for pvlib south orientation is 180. Here, azimuth is adapted for pvlib.
    if azimuth == 180:
        azimuth_pvlib = 0
    else:
        azimuth_pvlib = azimuth + 180

    # Obtain weather data from PVGIS database.

    weather = iotools.get_pvgis_hourly(
        coor_pv.Latitude,
        coor_pv.Longitude,
        start=start_year,
        end=end_year,
        url="https://re.jrc.ec.europa.eu/api/v5_2/",
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        map_variables=True,
    )[0]

    for counter_correction in range(len(weather.index)):
        if (
            weather.index.year[counter_correction] != start_year
            or weather.index.year[counter_correction] != end_year
        ):
            weather.drop(weather.index[counter_correction])
    weather.index.name = "utc_time"
    weather.index = weather.index.floor(freq="H")

    # Irradiance Calculations
    # Solarposition from location data.
    solpos = solarposition.get_solarposition(
        time=weather.index,
        latitude=coor_pv.Latitude,
        longitude=coor_pv.Longitude,
        altitude=coor_pv.Altitude,
        temperature=weather["temp_air"],
        pressure=atmosphere.alt2pres(coor_pv.Altitude),
    )
    # Some interesting atmospheric properties are computed to enhance the calculation of the exact irradiance reaching the panel.
    dni_extra = irradiance.get_extra_radiation(
        weather.index
    )  # Extra-atmospheric irradiance.
    airmass = atmosphere.get_relative_airmass(
        solpos["apparent_zenith"]
    )  # Air mass passing by each ray.
    pressure = atmosphere.alt2pres(coor_pv.Altitude)
    am_abs = atmosphere.get_absolute_airmass(airmass, pressure)
    aoi = irradiance.aoi(
        tilt,
        azimuth_pvlib,
        solpos["apparent_zenith"],
        solpos["azimuth"],
    )

    # Keep radiance data in the dataframe call total_irradiance. Here we have all the components of the irradiance in the plane-of-array (tilted module).

    poa_global = (
        weather["poa_direct"]
        + weather["poa_ground_diffuse"]
        + weather["poa_sky_diffuse"]
    )
    poa_diffuse = weather["poa_ground_diffuse"] + weather["poa_sky_diffuse"]
    total_irradiance = pd.DataFrame(
        {
            "poa_global": poa_global,
            "poa_direct": weather.poa_direct,
            "poa_diffuse": poa_diffuse,
            "poa_sky_diffuse": weather.poa_sky_diffuse,
            "poa_ground_diffuse": weather.poa_ground_diffuse,
        }
    )
    total_irradiance.index = weather.index

    #     Here we use an inverse transposition model to get ghi,dni and dhi from poa irradiances.
    horizontal_irradiance = irradiance.gti_dirint(
        total_irradiance["poa_global"],
        aoi,
        solpos["apparent_zenith"],
        solpos["azimuth"],
        weather.index,
        tilt,
        azimuth_pvlib,
        pressure,
        use_delta_kt_prime=True,
        temp_dew=None,
        albedo=0.2,
        model="perez",
        model_perez="allsitescomposite1990",
        calculate_gt_90=True,
        max_iterations=30,
    )

    # Hourly cell temperature

    cell_temperature = temperature.sapm_cell(
        total_irradiance["poa_global"],
        weather["temp_air"],
        weather["wind_speed"],
        **temperature_model_parameters,
    )

    # Here, shading losses between rows are applied and shading factors are calculated between two rows and between the panel and the ground.
    (shading_factors_output, diffuse_corrector) = shading_factors(
        tilt, azimuth, inter_row_space, dict_pv, coor_pv, total_irradiance.index
    )
    total_irradiance["poa_direct"] = total_irradiance["poa_direct"] * (
        1 - shading_factors_output["PV_Shading_Factor"]
    )

    #        Effective irradiance taking in account optical losses.
    effective_irradiance = pvsystem.sapm_effective_irradiance(
        total_irradiance["poa_direct"],
        total_irradiance["poa_diffuse"],
        am_abs,
        aoi,
        module,
    )

    # Calculate dc and ac power generated for one module.

    info_pv_dc = pvsystem.sapm(effective_irradiance, cell_temperature, module)
    # info_pv_ac = inverter.sandia(info_pv_dc["v_mp"], info_pv_dc["p_mp"], inverter_selected)

    # Based on the geometry of the system, a calculation will be performed to determine the number of panels that can be installed and, as a result, the total power they will generate collectively.

    pv_key = ["Length", "Width", "Stacked"]
    for key in pv_key:
        if (key in dict_pv) == False:
            raise KeyError("No esta la clave " + key)

    
    """
    portrait                   landscape
    
      w_pv                           w_pv                                          
    - - - - - --              -----------------
    -          -              -               -
    -          -              -               -
    -          -   l_pv       -               -  l_pv
    -          -              -               -
    -          -              -----------------
    - - - - - -- 
                                                                                                                                                                                             
                                                                        
    """                                                                 
    w_pv = dict_pv["Width"]
    l_pv = dict_pv["Length"]

    l_ground = ground_dim.Length
    w_ground = ground_dim.Width
    tilt_rad = math.radians(tilt)
    distance_condition = inter_row_space
    if (
        distance_condition < 1.25
    ):  # A minimum distance of 1.25 meters between rows is always set due to design conditions.
        space_final = 1.25
    else:
        space_final = inter_row_space

    correction_per_row = 4  # 4 meters
    pv_per_row = ((w_ground - correction_per_row) / w_pv) * dict_pv["Stacked"]
    n_rows = int(l_ground / (space_final)) + 1
    n_pv = n_rows * pv_per_row
    MW2W = 1000000  # Conversion from MW (megawatts) to W (watts)

    if start_year == end_year:
        n_years = 1
    else:
        n_years = (end_year - start_year) + 1

    # P_hourly = info_pv_dc.p_mp * n_pv * dict_pv['performance_factor']

    # ---------------------------------------ADAPTATION TO DSSC POWER-------------------------------

    P_hourly_conv = info_pv_dc.p_mp * n_pv * 1
    P_hourly_dssc = info_pv_dc.p_mp * n_pv * 0.6
    P_dc_conv = P_hourly_conv.sum() / (MW2W * n_years)
    P_dc_dssc = P_hourly_dssc.sum() / (MW2W * n_years)

    # --------------------------------- --------Total Power Output Calculation for Solar Panel Array-----------------------------------------

    results = pd.DataFrame(
        {"Potencia": P_hourly_conv, "real_ghi": horizontal_irradiance["ghi"]}
    )
    results.index = total_irradiance.index
    print("Potencia Conv (MWh): ", P_dc_conv)
    print("Potencia Dssc (MWh): ", P_dc_dssc)
    print("Número de paneles: ", n_pv)
    print("Número de hileras: ", n_rows)
    print("Densidad de paneles (%):", (l_pv / inter_row_space) * 100)
    print("Hectáreas:", (l_ground * w_ground) / 10000)
    zenith_rad = (solpos.zenith) * (math.pi / 180)
    cos_zenith = np.cos(zenith_rad)
    # Total Diffuse irradiance
    dhi_col = horizontal_irradiance.dhi
    # Total Direct Irradiance
    dni_col = horizontal_irradiance.dni

    # ---------------------------------------DIFFUSSE CORRECTOR-------------------------------------------------
    df_keeper = pd.DataFrame()
    df_keeper.index = total_irradiance.index
    A = np.array([[]] * len(total_irradiance) + [[1]])[:-1]
    df_keeper["corrector"] = A
    for index, row in diffuse_corrector.iterrows():
        current_selected_index = (df_keeper.index.month == index.month) & (
            df_keeper.index.hour == index.hour
        )
        df_keeper[current_selected_index] = row.diffuse_corrector

    # ----------------------------------------------------------------------------------
    # Direct Irradiance decomposition
    dni_shaded_col = dni_col * shading_factors_output.Ground_Shading_Factor
    dni_clear_col = dni_col * (1 - shading_factors_output.Ground_Shading_Factor)
    dhi_shaded_col_placa = (dhi_col * shading_factors_output.Ground_Shading_Factor) * (
        1 - df_keeper.corrector
    )
    dhi_shaded_col = (
        dhi_col * shading_factors_output.Ground_Shading_Factor
    ) * df_keeper.corrector
    dhi_clear_col = dhi_col * (1 - shading_factors_output.Ground_Shading_Factor)

    #"True" if you want to obtain the total hourly irradiance that reaches
    #the ground in its spectral decomposition / 
    #"False" to only obtain the total annual electric power output.
    
    if is_spectrum:
        A = np.array([[]] * len(results) + [[1]])[:-1]
        # results['clear_spectrum'] = A
        # results['shaded_spectrum'] = A
        # results['ghi_spectrum_col'] = A
        # results['dhi_spectrum_col'] = A
        # results['dni_spectrum_col'] = A
        # results['open_field_spectrum'] = A

        #         ----------------------------------CHANGES -----------------------------------
        results["ghi_spectrum_col_conv"] = A
        results["dhi_spectrum_col_conv"] = A
        results["dni_spectrum_col_conv"] = A

        results["ghi_spectrum_col_dssc"] = A
        results["dhi_spectrum_col_dssc"] = A
        results["dni_spectrum_col_dssc"] = A

        results["open_field_spectrum"] = A

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

        results["dni_spectrum_col_conv"] = (
            spectral_pv_trans["clear_spectrum"] * dni_clear_col
            + spectral_pv_trans["shaded_spectrum_conv"] * dni_shaded_col
        )
        results["dhi_spectrum_col_conv"] = (
            spectral_pv_trans["clear_spectrum"] * dhi_clear_col
            + spectral_pv_trans["clear_spectrum"] * dhi_shaded_col
            + spectral_pv_trans["shaded_spectrum_conv"] * dhi_shaded_col_placa
        )
        results["ghi_spectrum_col_conv"] = (
            results["dni_spectrum_col_conv"] * cos_zenith
            + results["dhi_spectrum_col_conv"]
        )

        results["dni_spectrum_col_dssc"] = (
            spectral_pv_trans["clear_spectrum"] * dni_clear_col
            + spectral_pv_trans["shaded_spectrum_dssc"] * dni_shaded_col
        )
        results["dhi_spectrum_col_dssc"] = (
            spectral_pv_trans["clear_spectrum"] * dhi_clear_col
            + spectral_pv_trans["clear_spectrum"] * dhi_shaded_col
            + spectral_pv_trans["shaded_spectrum_dssc"] * dhi_shaded_col_placa
        )
        # results['dhi_spectrum_col_dssc'] = spectral_pv_trans['clear_spectrum'] * dhi_col
        results["ghi_spectrum_col_dssc"] = (
            results["dni_spectrum_col_dssc"] * cos_zenith
            + results["dhi_spectrum_col_dssc"]
        )

        results["open_field_spectrum"] = (
            spectral_pv_trans["clear_spectrum"] * horizontal_irradiance["ghi"]
        )

    # Outputs
    return P_dc_conv, P_dc_dssc, results
    # return results
    
    
    
    
    
    
    
    """ 
    -----------------------AZIMUTH PVLIB TO PVFACTORS-----------------------
    """
def azimuth_conv(azimuth):
        
    if azimuth >= 0 and azimuth <= 90:
        azimuth_pvfactors = azimuth + ((90 - azimuth) * 2)
    elif azimuth > 90 and azimuth <= 180:
        azimuth_pvfactors = azimuth - ((azimuth - 90) * 2)
    elif azimuth < 0 and azimuth >= -90:
        azimuth_pvfactors = (azimuth + ((90 - azimuth) * 2)) - 360
    elif azimuth > -180 and azimuth < -90:
        azimuth_pvfactors = (azimuth - ((azimuth - 90) * 2)) - 360
    else:
        raise ArithmeticError(
            "It is mandatory to insert a azimuth value between -180 y 180"
        )
    return azimuth_pvfactors

