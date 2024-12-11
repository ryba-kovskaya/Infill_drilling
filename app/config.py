import pandas as pd

# Названия колонок в файле МЭР
columns_name = {'Дата': 'date',
                '№ скважины': 'well_number',
                'Месторождение': 'field',
                'Объект': 'object',
                'Объекты работы': 'objects',
                'Характер работы': 'work_marker',
                'Состояние': 'well_status',
                'Дебит нефти за последний месяц, т/сут': 'Qo_rate',
                'Дебит нефти (ТР), т/сут': 'Qo_rate_TR',
                'Дебит жидкости за последний месяц, т/сут': 'Ql_rate',
                'Дебит жидкости (ТР), м3/сут': 'Ql_rate_TR',
                'Добыча нефти за посл.месяц, т': 'Qo',
                'Добыча жидкости за посл.месяц, т': 'Ql',
                'Обводненность за посл.месяц, % (вес)': 'water_cut',
                'Обводненность (ТР), % (объём)': 'water_cut_TR',
                'Приемистость (ТР), м3/сут': 'Winj_rate_TR',
                'Приемистость за последний месяц, м3/сут': 'Winj_rate',
                'Закачка за посл.месяц, м3': 'Winj',
                'Время работы, часы': 'time_work',
                'Время работы в добыче, часы': 'time_work_prod',
                'Время работы под закачкой, часы': 'time_work_inj',
                'Забойное давление (ТР), атм': 'P_well',
                'Пластовое давление (ТР), атм': 'P_reservoir',
                'Координата X': 'T1_x_geo',
                'Координата Y': 'T1_y_geo',
                'Координата забоя Х (по траектории)': 'T3_x_geo',
                'Координата забоя Y (по траектории)': 'T3_y_geo'}

# Характер работы скважин
dict_work_marker = {"НЕФ": "prod",
                    "НАГ": "inj"}

# Названия колонок в файле ГФХ
gpch_column_name = {'Месторождение': {'field': 'str'},
                    'Тип данных': {'data_type': 'str'},
                    'Объект': {'object': 'str'},
                    'Пласт': {'pool': 'str'},
                    'Район': {'area': 'str'},
                    'Средняя глубина залегания кровли': {'top_depth': 'float'},
                    'Абсолютная отметка ВНК': {'oil-water_contact_depth': 'float'},
                    'Абсолютная отметка ГНК': {'gas-oil_contact_depth': 'float'},
                    'Абсолютная отметка ГВК': {'gas-water_contact_depth': 'float'},
                    'Тип залежи': {'type_pool': 'str'},
                    'Тип коллектора': {'reservoir_type': 'str'},
                    'Площадь нефтеносности': {'oil_productive_area': 'float'},
                    'Площадь газоносности': {'gas_productive_area': 'float'},
                    'Средняя общая толщина': {'Avg_total_h': 'float'},
                    'Средняя эффективная нефтенасыщенная толщина': {'eff_oil_h': 'float'},
                    'Средняя эффективная газонасыщенная толщина': {'eff_gas_h': 'float'},
                    'Средняя эффективная водонасыщенная толщина': {'eff_water_h': 'float'},
                    'Коэффициент пористости': {'porosity': 'float'},
                    'Коэффициент нефтенасыщенности ЧНЗ': {'oil_saturation_oil_zone': 'float'},
                    'Коэффициент нефтенасыщенности ВНЗ': {'oil_saturation_water_oil_zone': 'float'},
                    'Коэффициент нефтенасыщенности пласта': {'total_oil_saturation': 'float'},
                    'Коэффициент газонасыщенности пласта': {'total_gas_saturation': 'float'},
                    'Проницаемость': {'permeability': 'float'},
                    'Коэффициент песчанистости': {'gross_sand_ratio ': 'float'},
                    'Расчлененность': {'stratification_factor': 'float'},
                    'Начальная пластовая температура': {'init_temperature': 'float'},
                    'Начальное пластовое давление': {'init_pressure': 'float'},
                    'Вязкость нефти в пластовых условиях': {'oil_viscosity_in_situ': 'float'},
                    'Плотность нефти в пластовых условиях': {'oil_density_in_situ': 'float'},
                    'Плотность нефти в поверхностных условиях': {'oil_density_at_surf': 'float'},
                    'Объемный коэффициент нефти': {'Bo': 'float'},
                    'Содержание серы в нефти': {'sulphur_content': 'float'},
                    'Содержание парафина в нефти': {'paraffin_content': 'float'},
                    'Давление насыщения нефти газом': {'bubble_point_pressure': 'float'},
                    'Газосодержание': {'gas_oil_ratio': 'float'},
                    'Давление начала конденсации': {'dewpoint_pressure': 'float'},
                    'Плотность конденсата в стандартных условиях': {'condensate_density_in_st': 'float'},
                    'Вязкость конденсата в стандартных условиях': {'condensate_viscosity_in_st': 'float'},
                    'Потенциальное содержание стабильного конденсата в газе (С5+)': {
                        'stabilized_condensate_content_gas': 'float'},
                    'Содержание сероводорода': {'hydrogen_sulfide_content': 'float'},
                    'Вязкость газа в пластовых условиях': {'gas_viscosity_in_situ': 'float'},
                    'Плотность газа в пластовых условиях': {'gas_density_in_situ': 'float'},
                    'Коэффициент сверхсжимаемости газа': {'z_factor': 'float'},
                    'Вязкость воды в пластовых условиях': {'water_viscosity_in_situ': 'float'},
                    'Плотность воды в поверхностных условиях': {'water_density_at_surf': 'float'},
                    'Сжимаемость': {'compressibility_del': 'str'},
                    'нефти': {'oil_compressibility': 'float'},
                    'воды': {'water_compressibility': 'float'},
                    'породы': {'formation_compressibility': 'float'},
                    'Коэффициент вытеснения (водой)': {'water_flood_displacement_efficiency': 'float'},
                    'Коэффициент вытеснения (газом)': {'gas_flood_displacement_efficiency': 'float'},
                    'Коэффициент продуктивности': {'productivity_factor': 'float'},
                    'Коэффициенты фильтрационных сопротивлений:': {'flow_coefficient_del': 'str'},
                    'А': {'flow_coefficient_A': 'float'},
                    'В': {'flow_coefficient_B': 'float'}}

# Наполнение data_wells DataFrame
sample_data_wells = pd.DataFrame(columns=[

    # general_info:
    'well_number',  # [1] номер скважины
    'work_marker',  # [2] характер работы [prod, inj]
    'well_status',  # [3] состояние
    'well_type',  # [4] тип скважины [vertical, horizontal]

    # production_params: (все приводятся за последний рабочий месяц)
    'date',  # [5] последняя дата работы
    'Qo_rate',  # [6] дебит нефти, т/сут
    'Qo_rate_TR',  # [7] дебит нефти ТР, т/сут
    'init_Qo_rate_TR',  # [8] стартовый дебит жидкости ТР, т/сут
    'Ql_rate',  # [9] стартовый дебит жидкости, т/сут
    'Ql_rate_TR',  # [10] дебит жидкости ТР, т/сут
    'init_Ql_rate_TR',  # [11] стартовый дебит жидкости ТР, т/сут
    'Qo',  # [12] добыча жидкости, т
    'Ql',  # [13] добыча жидкости, т
    'water_cut',  # [14] обводненность, %
    'water_cut_TR',  # [15] обводненность ТР, %
    'init_water_cut',  # [16] стартовая обводненность, %
    'init_water_cut_TR',  # [17] стартовая обводненность ТР, %
    'Winj_rate_TR',  # [18] приемистость ТР, м3/сут
    'Winj_rate',  # [19] приемистость, м3/сут
    'Winj',  # [20] закачка, м3
    'time_work',  # [21] время работы, часы
    'time_work_prod',  # [22] время работы в добыче, часы
    'time_work_inj',  # [23] время работы в закачке, часы
    'P_well',  # [24] забойное давление, атм
    'init_P_well_prod',  # [25] запускное забойное давление добывающей скважины, атм
    'init_P_well_inj',  # [26] запускное забойное давление нагнетательной скважины, атм
    'no_work_time',  # [27] количество месяцев в простое
    'Qo_cumsum',  # [27] накопленная добыча нефти, т
    'Winj_cumsum',  # [29] накопленная закачка, м3
    'init_Qo_rate',  # [30] средний запускной дебит нефти за 6 месяцев со старта, м3/сут
    'init_Ql_rate',  # [31] средний запускной дебит жидкости за 6 месяцев со старта, м3/сут
    'r_eff_not_norm',  # [32] эффективный радиус скважины рассчитанный на основе порового объема, м
    'r_eff_voronoy',  # [33] эффективный радиус через площадь ячейки вороного, м
    'r_eff',  # [34] эффективный радиус нормированный на ячейку вороного, м

    # reservoir_params:
    'P_reservoir',  # [35] пластовое давление ТР, атм
    'init_P_reservoir_prod',  # [36] стартовое пластовое давление ТР, атм
    # снятые с карт
    'NNT',  # [37] нефтенасыщенная толщина, м
    'm',  # [38] пористость, д.ед
    'So',  # [39] начальная нефтенасыщенность, д.ед
    'permeability',  # [40] проницаемость c карты, мД
    'permeability_fact',  # [41] проницаемость обратным счетом через РБ, мД

    # location:
    'T1_x_geo',  # [42] географическая координата X точки T1
    'T1_y_geo',  # [43] географическая координата Y точки T1
    'T3_x_geo',  # [44] географическая координата X точки T3
    'T3_y_geo',  # [45] географическая координата Y точки T3
    'length_geo',  # [46] длина ствола скважины T1-T3, м
    'T1_x_pix',  # [47] координата в пикселях X точки T1
    'T1_y_pix',  # [48] координата в пикселях Y точки T1
    'T3_x_pix',  # [49] координата в пикселях X точки T3
    'T3_y_pix',  # [50] координата в пикселях Y точки T3
    'length_pix',  # [51] длина ствола скважины T1-T3, пиксели
    'azimuth',  # [52] азимут скважины, градусы

    # Shapely геометрия:
    'POINT_T1_geo',  # [53] тип POIT для географической точки T1
    'POINT_T3_geo',  # [54] тип POIT для географической точки T3
    'LINESTRING_geo',  # [55] LINESTRING для ствола скважины в географических координатах
    'POINT_T1_pix',  # [56] тип POIT для точки T1 в пикселях
    'POINT_T3_pix',  # [57] тип POIT для точки T3 в пикселях
    'LINESTRING_pix'])  # [58] LINESTRING для ствола скважины в пикселях
