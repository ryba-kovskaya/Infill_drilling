import os
import numpy as np
from loguru import logger
from osgeo import gdal, ogr

from .maps import (Map, read_array, read_raster, get_map_reservoir_score, get_map_potential_score,
                   get_map_risk_score, get_map_opportunity_index)
from ..input_output.input import create_shapely_types
from ..well_active_zones import get_value_map
from ..ranking_drilling.one_phase_model import get_sw


@logger.catch
def mapping(maps_directory, data_wells, dict_properties, **kwargs):
    """Загрузка, подготовка и расчет всех необходимых карт"""
    # Инициализация параметров загрузки
    default_size_pixel = kwargs['default_size_pixel']
    radius_interpolate = kwargs['radius_interpolate']
    accounting_GS = kwargs['accounting_GS']

    logger.info(f"path: {maps_directory}")
    content = os.listdir(path=maps_directory)
    if content:
        logger.info(f"count of maps: {len(content)}")
    else:
        raise logger.critical("no maps!")

    logger.info(f"Загрузка карт из папки: {maps_directory}")
    maps = maps_load_directory(maps_directory)

    # Поиск наименьшего размера карты и размера пикселя, если он None при загрузке
    dst_geo_transform, dst_projection, shape = get_final_resolution(maps, default_size_pixel)

    logger.info(f"Построение карт на основе дискретных значений")
    maps = maps + maps_load_df(data_wells, dst_geo_transform, shape, accounting_GS, radius_interpolate)

    logger.info(f"Преобразование карт к единому размеру и сетке")
    maps = list(map(lambda raster: raster.resize(dst_geo_transform, dst_projection, shape), maps))

    logger.info(f"Расчет оценочных карт")
    maps = maps + calculate_score_maps(maps, dict_properties)

    logger.info(f"Перевод координат скважин в пиксельные")
    data_wells['T1_x_pix'], data_wells['T1_y_pix'] = maps[-1].convert_coord_to_pix(
        (data_wells["T1_x_geo"].to_numpy(), data_wells["T1_y_geo"].to_numpy()))
    data_wells['T3_x_pix'], data_wells['T3_y_pix'] = maps[-1].convert_coord_to_pix(
        (data_wells["T3_x_geo"].to_numpy(), data_wells["T3_y_geo"].to_numpy()))
    data_wells['length_pix'] = np.sqrt(np.power(data_wells.T3_x_pix - data_wells.T1_x_pix, 2)
                                       + np.power(data_wells.T3_y_pix - data_wells.T1_y_pix, 2))
    # расчет Shapely объектов
    df_shapely = create_shapely_types(data_wells, list_names=['T1_x_pix', 'T1_y_pix', 'T3_x_pix', 'T3_y_pix'])
    data_wells[['POINT_T1_pix', 'POINT_T3_pix', 'LINESTRING_pix']] = df_shapely

    logger.info(f"Запись значений с карт для текущего фонда")
    # с карт снимаем значения eff_h (NNT), m, So
    type_map_list = list(map(lambda raster: raster.type_map, maps))
    dict_column_map = {'NNT': "NNT", 'm': "porosity", 'So': "initial_oil_saturation",
                       'permeability': "permeability"}
    for column_key in dict_column_map.keys():
        data_wells[column_key] = data_wells.apply(
            lambda row: get_value_map(row['well_type'], row['T1_x_pix'], row['T1_y_pix'], row['T3_x_pix'],
                                      row['T3_y_pix'], row['length_pix'],
                                      raster=maps[type_map_list.index(dict_column_map[column_key])]), axis=1)
    return maps, data_wells


def maps_load_directory(maps_directory):
    maps = []

    logger.info(f"Загрузка карты ННТ")
    try:
        maps.append(read_raster(f'{maps_directory}/NNT.grd'))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой ННТ: NNT.grd")

    logger.info(f"Загрузка карты проницаемости")
    try:
        maps.append(read_raster(f'{maps_directory}/permeability.grd'))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой проницаемости: permeability.grd")

    logger.info(f"Загрузка карты ОИЗ")
    try:
        maps.append(read_raster(f'{maps_directory}/residual_recoverable_reserves.grd'))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой ОИЗ: residual_recoverable_reserves.grd")

    logger.info(f"Загрузка карты изобар")
    try:
        maps.append(read_raster(f'{maps_directory}/pressure.grd'))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой изобар: pressure.grd")

    logger.info(f"Загрузка карты начальной нефтенасыщенности")
    try:
        maps.append(read_raster(f'{maps_directory}/initial_oil_saturation.grd'))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой изобар: initial_oil_saturation.grd")

    logger.info(f"Загрузка карты пористости")
    try:
        maps.append(read_raster(f'{maps_directory}/porosity.grd'))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой пористости: porosity.grd")

    return maps


def maps_load_df(data_wells, dst_geo_transform, shape, accounting_GS, radius_interpolate):
    maps = []
    #  Загрузка карт из "МЭР"
    logger.info(f"Загрузка карты обводненности на основе выгрузки МЭР")
    maps.append(read_array(data_wells, name_column_map="water_cut", type_map="water_cut",
                           geo_transform=dst_geo_transform, size=shape,
                           accounting_GS=accounting_GS, radius=radius_interpolate))

    logger.info(f"Загрузка карты последних дебитов нефти на основе выгрузки МЭР")
    maps.append(read_array(data_wells, name_column_map="Qo_rate", type_map="last_rate_oil",
                           geo_transform=dst_geo_transform, size=shape,
                           accounting_GS=accounting_GS, radius=radius_interpolate))

    logger.info(f"Загрузка карты стартовых дебитов нефти на основе выгрузки МЭР")
    maps.append(read_array(data_wells, name_column_map="init_Qo_rate", type_map="init_rate_oil",
                           geo_transform=dst_geo_transform, size=shape,
                           accounting_GS=accounting_GS, radius=radius_interpolate))
    return maps


def get_final_resolution(list_rasters, pixel_sizes):
    """Поиск наименьшего размера карты, который станет целевым для расчета

    list_rasters - список карт
    pixel_sizes - шаг сетки int/None (по-умолчанию 50/поиск наименьшего шага среди сеток)

    return: geo_transform, projection, shape
    """
    data_list = list(map(lambda raster: raster.data, list_rasters))
    geo_transform_list = list(map(lambda raster: raster.geo_transform, list_rasters))
    projection_list = list(map(lambda raster: raster.projection, list_rasters))

    min_x = max(list(map(lambda geo_transform: geo_transform[0], geo_transform_list)))
    max_x = min(list(map(lambda geo_transform, data: geo_transform[0] + geo_transform[1] * data.shape[1],
                         geo_transform_list, data_list)))
    min_y = max(list(map(lambda geo_transform, data: geo_transform[3] + geo_transform[5] * data.shape[0],
                         geo_transform_list, data_list)))
    max_y = min(list(map(lambda geo_transform: geo_transform[3], geo_transform_list)))

    if not pixel_sizes:
        pixel_size_x = round(min(list(map(lambda geo_transform: geo_transform[1], geo_transform_list))) / 5) * 5
        pixel_size_y = round(min(list(map(lambda geo_transform: abs(geo_transform[5]), geo_transform_list))) / 5) * 5
        pixel_sizes = min(pixel_size_x, pixel_size_y)

    cols = int((max_x - min_x) / pixel_sizes)
    rows = int((max_y - min_y) / pixel_sizes)
    shape = (rows, cols)

    dst_geo_transform = (min_x, pixel_sizes, 0, max_y, 0, -pixel_sizes)
    dst_projection = max(set(projection_list), key=projection_list.count)

    return dst_geo_transform, dst_projection, shape


def calculate_score_maps(maps, dict_properties):
    """
    Расчет оценочных карт для дальнейшего анализа
    Parameters
    ----------
    maps - обязательный набор карт списком (порядок не важен):
        [NNT, permeability, residual_recoverable_reserves, pressure, initial_oil_saturation,
        water_cut, last_rate_oil, init_rate_oil]
    dict_properties - ГФХ пласта
    Returns
    -------
    maps - [reservoir_score, potential_score, risk_score, opportunity_index]
    """
    type_maps_list = list(map(lambda raster: raster.type_map, maps))

    # инициализация всех необходимых карт из списка
    map_NNT = maps[type_maps_list.index("NNT")]
    map_permeability = maps[type_maps_list.index("permeability")]
    map_residual_recoverable_reserves = maps[type_maps_list.index("residual_recoverable_reserves")]
    map_pressure = maps[type_maps_list.index("pressure")]
    map_initial_oil_saturation = maps[type_maps_list.index("initial_oil_saturation")]

    map_water_cut = maps[type_maps_list.index("water_cut")]
    map_last_rate_oil = maps[type_maps_list.index("last_rate_oil")]
    map_init_rate_oil = maps[type_maps_list.index("init_rate_oil")]

    logger.info("Расчет карты оценки качества коллектора")
    map_reservoir_score = get_map_reservoir_score(map_NNT, map_permeability)

    logger.info("Расчет карты оценки потенциала")
    map_potential_score = get_map_potential_score(map_residual_recoverable_reserves,
                                                  map_last_rate_oil,
                                                  map_init_rate_oil)
    logger.info("Расчет карты оценки риска")
    init_pressure = dict_properties['P_init']
    if init_pressure == 0:
        init_pressure = np.max(map_pressure.data)
    map_risk_score = get_map_risk_score(map_water_cut, map_initial_oil_saturation, map_pressure,
                                        init_pressure=init_pressure)
    logger.info("Расчет карты индекса возможностей")
    map_opportunity_index = get_map_opportunity_index(map_reservoir_score, map_potential_score, map_risk_score)
    # где высокая обводненность opportunity_index = 0
    # !!! Может не 0, а 0.01 или что-то такое, чтобы не было дыр в карте
    map_opportunity_index.data[(map_water_cut.data > 99.5)] = 0
    # где нет толщин и давления opportunity_index = 0
    map_opportunity_index.data[(map_NNT.data == 0) & (map_pressure.data == 0)] = 0

    return [map_reservoir_score, map_potential_score, map_risk_score, map_opportunity_index]


def apply_wells_mask(base_map, data_wells):
    """
    Создание по карте области исключения (маски) на основе действующего фонда
    + как опция в будущем учет также проектного фонда
    Parameters
    ----------
    base_map - карта, на основе которой будет отстроена маска
    data_wells - фрейм с параметрами добычи на последнюю дату работы для всех скважин

    Returns
    -------
    modified_map - карта с вырезанной зоной действующих скважин
    """
    logger.info("Расчет буфера вокруг скважин")
    union_buffer = active_well_outline(data_wells)

    logger.info("Создание маски буфера")
    if union_buffer:
        mask = create_mask_from_buffers(base_map, union_buffer)
    else:
        mask = np.empty(base_map.data.shape)

    logger.info("Бланкование карты согласно буферу")
    modified_map = cut_map_by_mask(base_map, mask, blank_value=0)

    return modified_map


def active_well_outline(df_wells):
    """
    Создание буфера вокруг действующих скважин
     Parameters
    ----------
    df_wells - DataFrame скважин с обязательными столбцами:
                [well type, T1_x_geo, T1_y_geo, T3_x_geo, T3_y_geo]
    buffer_radius - расстояние от скважин, на котором нельзя бурить // в перспективе замена на радиус дренирования,
     нагнетания с индивидуальным расчетом для каждой скважины

    Returns
    -------
    union_buffer - POLYGON
    """

    if df_wells.empty:
        logger.warning('Нет скважин для создания маски')
        return

    def create_buffer(row):
        # Создание геометрии для каждой скважины
        buffer_radius = row.r_eff  # радиус из строки
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(row.T1_x_geo, row.T1_y_geo)
        line.AddPoint(row.T3_x_geo, row.T3_y_geo)
        return line.Buffer(buffer_radius)

    # Создаём список буферов
    buffers = [create_buffer(row) for _, row in df_wells.iterrows()]

    # Создание пустой геометрии типа MultiPolygon для объединения
    merged_geometry = ogr.Geometry(ogr.wkbMultiPolygon)
    # Добавляем каждую геометрию в объединенную геометрию
    for geom in buffers:
        merged_geometry.AddGeometry(geom)
    # Объединяем все геометрии в одну
    union_buffer = merged_geometry.UnionCascaded()

    return union_buffer


def create_mask_from_buffers(base_map, buffer):
    """
    Функция создания маски из буфера в видe array
    Parameters
    ----------
    base_map - карта, с которой получаем geo_transform для определения сетки для карты с буфером
    buffer - буфер вокруг действующих скважин

    Returns
    -------
    mask - array
    """
    # Размер карты
    cols = base_map.data.shape[1]
    rows = base_map.data.shape[0]

    # Информация о трансформации
    transform = base_map.geo_transform
    projection = base_map.projection

    # Создаем временный растровый слой
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(projection)

    # Создаем векторный слой в памяти для буфера
    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('')  # Создаем временный источник данных
    mem_layer = mem_ds.CreateLayer('', None, ogr.wkbPolygon)  # Создаем слой в этом источнике данных
    feature = ogr.Feature(mem_layer.GetLayerDefn())  # Создание нового объекта
    feature.SetGeometry(buffer)  # Установка геометрии для объекта
    mem_layer.CreateFeature(feature)  # Добавление объекта в слой

    # Заполнение растра значениями по умолчанию
    band = dataset.GetRasterBand(1)  # первый слой
    band.SetNoDataValue(0)
    band.Fill(0)  # Установка всех значений пикселей в 0

    # Растеризация буферов
    gdal.RasterizeLayer(dataset, [1], mem_layer, burn_values=[1], options=['ALL_TOUCHED=TRUE'])

    # Чтение данных из растрового бэнда в массив
    mask = band.ReadAsArray()

    # Закрываем временные данные
    dataset, mem_ds = None, None

    return mask


def cut_map_by_mask(base_map, mask, blank_value=np.nan):
    """
    Обрезаем карту согласно маске (буферу вокруг действующих скважин)
    Parameters
    ----------
    map - карта для обрезки
    mask - маска буфера
    blank_value - значение бланка

    Returns
    -------
    modified_map - обрезанная карта
    """
    # Создаем копию данных карты и заменяем значения внутри маски
    modified_data = base_map.data.copy()
    modified_data[mask == 1] = blank_value

    # Создаем новый объект Map с модифицированными данными
    modified_map = Map(modified_data,
                       base_map.geo_transform,
                       base_map.projection,
                       base_map.type_map)

    return modified_map


def calculate_Sw(row, dict_parameters_coefficients):
    """

    Parameters
    ----------

    Returns
    -------

    """
    # Переопределим параметры из словаря
    kv_kh, Swc, Sor, Fw, m1, Fo, m2, Bw = (
        list(map(lambda name: dict_parameters_coefficients['default_well_params'][name],
                 ['kv_kh', 'Swc', 'Sor', 'Fw', 'm1', 'Fo', 'm2', 'Bw'])))

    reservoir_params = dict_parameters_coefficients['reservoir_params']
    reservoir_params['f_w'] = row['water_cut']
    fluid_params = dict_parameters_coefficients['fluid_params']

    # Выделение необходимых параметров пластовой жидкости
    mu_w, mu_o, Bo, c_o, c_w = list(map(lambda key: fluid_params[key], ['mu_w', 'mu_o', 'Bo', 'c_o', 'c_w']))
    # Выделение необходимых параметров пласта
    f_w, c_r = list(map(lambda key: reservoir_params[key], ['f_w', 'c_r']))

    if row.Winj_rate == 0:
        # Расчет насыщенности
        Sw = get_sw(mu_w, mu_o, Bo, Bw, f_w, Fw, m1, Fo, m2, Swc, Sor)
        return 1 - Sw - Sor
    else:
        return Sor