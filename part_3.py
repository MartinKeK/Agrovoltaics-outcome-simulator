
import yaml
import pcse
def get_agro_info(crop_name, PAR_results):
    crop_year_int = PAR_results.index[1].year
    crop_year_str = str(crop_year_int)

    if crop_name == "potato":
        crop_name = "potato"
        variety_name = "Potato_701"
        campaign_start_date = "xxxx-04-01"
        campaign_start_date = campaign_start_date.replace("xxxx", crop_year_str)
        emergence_date = "xxxx-04-30"
        emergence_date = emergence_date.replace("xxxx", crop_year_str)
        harvest_date = "xxxx-09-10"
        harvest_date = harvest_date.replace("xxxx", crop_year_str)
        max_duration = 160

    elif crop_name == "maize":
        crop_name = "maize"
        variety_name = "Grain_maize_201"
        campaign_start_date = "xxxx-04-01"
        campaign_start_date = campaign_start_date.replace("xxxx", crop_year_str)
        emergence_date = "xxxx-04-30"
        emergence_date = emergence_date.replace("xxxx", crop_year_str)
        harvest_date = "xxxx-11-20"
        harvest_date = harvest_date.replace("xxxx", crop_year_str)
        max_duration = 250

    elif crop_name == "sunflower":
        crop_name = "sunflower"
        variety_name = "Sunflower_1101"
        campaign_start_date = "xxxx-04-01"
        campaign_start_date = campaign_start_date.replace("xxxx", crop_year_str)
        emergence_date = "xxxx-04-20"
        emergence_date = emergence_date.replace("xxxx", crop_year_str)
        harvest_date = "xxxx-10-30"
        harvest_date = harvest_date.replace("xxxx", crop_year_str)
        max_duration = 180
    elif crop_name == "barley":
        crop_name = "barley"
        variety_name = "Spring_barley_301"
        campaign_start_date = "xxxx-03-01"
        campaign_start_date = campaign_start_date.replace("xxxx", crop_year_str)
        emergence_date = "xxxx-03-31"
        emergence_date = emergence_date.replace("xxxx", crop_year_str)
        harvest_date = "xxxx-09-15"
        harvest_date = harvest_date.replace("xxxx", crop_year_str)
        max_duration = 180
    else:
        raise KeyError("Crop Name invalid")
    # Here we define the agromanagement for the selected crop

    agro_yaml = """
    - {start}:
        CropCalendar:
            crop_name: {cname}
            variety_name: {vname}
            crop_start_date: {startdate}
            crop_start_type: emergence
            crop_end_date: {enddate}
            crop_end_type: harvest
            max_duration: {maxdur}
        TimedEvents: null
        StateEvents: null
    """.format(
        cname=crop_name,
        vname=variety_name,
        start=campaign_start_date,
        startdate=emergence_date,
        enddate=harvest_date,
        maxdur=max_duration,
    )
    return yaml.safe_load(agro_yaml)

def crop_simulation(agro_yaml, PAR_results, coor_pv):
    # Define Location

    # Define Crops Calendar
    
        # print(agro_yaml)

    # Weather
    wdp = pcse.db.NASAPowerWeatherDataProvider(
        latitude=coor_pv.Latitude, longitude=coor_pv.Longitude, force_update=False, ETmodel="PM"
    )

    # Parameter sets for crop, soil and site
    # Standard crop parameter library
    cropd = pcse.fileinput.YAMLCropDataProvider()
    # We don't need soil for potential production, so we use dummy values
    soild = pcse.util.DummySoilDataProvider()
    # Some site parameters
    sited = pcse.util.WOFOST72SiteDataProvider(WAV=50, CO2=360.0)

    # Retrieve all parameters in the form of a single object.
    # In order to see all parameters for the selected crop already, we
    # synchronise data provider cropd with the crop/variety:
    firstkey = list(agro_yaml[0])[0]
    cropcalendar = agro_yaml[0][firstkey]["CropCalendar"]
    cropd.set_active_crop(cropcalendar["crop_name"], cropcalendar["variety_name"])
    params = pcse.base.ParameterProvider(cropdata=cropd, sitedata=sited, soildata=soild)

    # Open-field calculation. This part is necessary to be able to compare the crop production in open field versus other shaded conditions.

    wdp_of = wdp
    for counter_of in range(len(PAR_results.index)):
        wdp_of(PAR_results.index[counter_of]).IRRAD = PAR_results[
            "PAR_absorption_tot_of"
        ][counter_of]

    wfpp = pcse.models.Wofost71_PP(params, wdp_of, agro_yaml)
    wfpp.run_till_terminate()
    summary_output = wfpp.get_summary_output()

    total_biomass_of = summary_output[0]["TAGP"]
    harvestable_product_of = summary_output[0]["TWSO"]

    # Under PV calculation

    #     wdp_pv = wdp
    #     for counter_pv in range(len(PAR_results.index)):
    #         wdp_pv(PAR_results.index[counter_pv]).IRRAD = PAR_results['PAR_absorption_tot'][counter_pv]

    #     wfpp = Wofost71_PP(params, wdp_pv, agro)
    #     wfpp.run_till_terminate()
    #     summary_output = wfpp.get_summary_output()

    #     total_biomass_pv = summary_output[0]['TAGP']
    #     harvestable_product_pv = summary_output[0]['TWSO']

    #     ---------------------------------------CAMBIOS-----------------------------------------
    # We input the calculated absorbed irradiance data from part 2 for both conventional cells and DSSCs.
    wdp_conv = wdp
    for counter_conv in range(len(PAR_results.index)):
        wdp_conv(PAR_results.index[counter_conv]).IRRAD = PAR_results[
            "PAR_absorption_tot_conv"
        ][counter_conv]

    wfpp = pcse.models.Wofost72_PP(
        params, wdp_conv, agro_yaml
    )  # Wofost72_PP (potencial production). It only takes into account the effect of irradiance, assuming that the other parameters that influence growth are always optimal. This way, we can compare the different results while only considering the irradiance, which is what is being investigated in this case.
    wfpp.run_till_terminate()
    summary_output = wfpp.get_summary_output()

    total_biomass_conv = summary_output[0]["TAGP"]
    harvestable_product_conv = summary_output[0]["TWSO"]

    wdp_dssc = wdp
    for counter_dssc in range(len(PAR_results.index)):
        wdp_dssc(PAR_results.index[counter_dssc]).IRRAD = PAR_results[
            "PAR_absorption_tot_dssc"
        ][counter_dssc]

    wfpp = pcse.models.Wofost72_PP(params, wdp_dssc, agro_yaml)
    wfpp.run_till_terminate()
    summary_output = wfpp.get_summary_output()

    total_biomass_dssc = summary_output[0]["TAGP"]
    harvestable_product_dssc = summary_output[0]["TWSO"]

    # ------------------------------------------------------------------------------------------

    return harvestable_product_of, harvestable_product_conv, harvestable_product_dssc
