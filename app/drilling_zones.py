import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from loguru import logger
from osgeo import gdal, ogr
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

from map import Map
from local_parameters import paths
from input_output import load_wells_data


def calculate_zones(maps):
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    # инициализация всех необходимых карт
    map_NNT = maps[type_map_list.index("NNT")]
    map_permeability = maps[type_map_list.index("permeability")]
    map_residual_recoverable_reserves = maps[type_map_list.index("residual_recoverable_reserves")]
    map_pressure = maps[type_map_list.index("pressure")]
    map_initial_oil_saturation = maps[type_map_list.index("initial_oil_saturation")]

    map_water_cut = maps[type_map_list.index("water_cut")]
    map_last_rate_oil = maps[type_map_list.index("last_rate_oil")]
    map_init_rate_oil = maps[type_map_list.index("init_rate_oil")]

    logger.info("Расчет карты оценки пласта")
    map_reservoir_score = reservoir_score(map_NNT, map_permeability)

    logger.info("Расчет карты оценки показателей разработки")
    map_potential_score = potential_score(map_residual_recoverable_reserves, map_pressure,
                                          map_last_rate_oil, map_init_rate_oil)

    logger.info("Расчет карты оценки проблем")
    map_risk_score = risk_score(map_water_cut, map_initial_oil_saturation)

    logger.info("Расчет карты индекса возможностей")
    map_opportunity_index = opportunity_index(map_reservoir_score, map_potential_score, map_risk_score)

    # где нет толщин, проницаемости и давления opportunity_index = 0
    map_opportunity_index.data[(map_NNT.data == 0) & (map_permeability.data == 0) & (map_pressure.data == 0)] = 0

    map_opportunity_index.save_img(f"{save_directory}/map_opportunity_index.png", data_wells)
    map_opportunity_index.save_grd_file(f"{save_directory}/opportunity_index.grd")

    logger.info("Создание по карте области исключения (маски) на основе действующего фонда")
    modified_map_opportunity_index = apply_wells_mask(map_opportunity_index, data_wells)

    modified_map_opportunity_index.save_img(f"{save_directory}/cut_map_opportunity_index.png", data_wells)
    modified_map_opportunity_index.save_grd_file(f"{save_directory}/cut_map_opportunity_index.grd")

    logger.info("Кластеризация зон")
    dict_zones = clusterization_zones(modified_map_opportunity_index, epsilon=15, min_samples=50, percent_low=80)
    map_opportunity_index.save_img(f"{save_directory}/map_opportunity_index_with_zones.png",
                                   data_wells, dict_zones)
    return dict_zones


def reservoir_score(map_NNT, map_permeability) -> Map:
    """
    Оценка пласта
    -------
    Map(type_map=reservoir_score)
    """
    norm_map_NNT = map_NNT.normalize_data()
    norm_map_permeability = map_permeability.normalize_data()

    data_reservoir_score = (norm_map_NNT.data + norm_map_permeability.data) / 2

    map_reservoir_score = Map(data_reservoir_score,
                              norm_map_NNT.geo_transform,
                              norm_map_NNT.projection,
                              "reservoir_score")
    map_reservoir_score.save_img(f"{save_directory}/map_reservoir_score.png", data_wells)
    return map_reservoir_score


def potential_score(map_residual_recoverable_reserves, map_pressure, map_last_rate_oil, map_init_rate_oil) -> Map:
    """
    Оценка показателей разработки
    -------
    Map(type_map=potential_score)
    """
    P_init = 40 * 9.87  # атм для Крайнего Ю1

    map_last_rate_oil.data = np.nan_to_num(map_last_rate_oil.data)
    map_init_rate_oil.data = np.nan_to_num(map_init_rate_oil.data)
    norm_last_rate_oil = map_last_rate_oil.normalize_data()
    norm_init_rate_oil = map_init_rate_oil.normalize_data()

    norm_residual_recoverable_reserves = map_residual_recoverable_reserves.normalize_data()
    map_delta_P = Map(P_init - map_pressure.data, map_pressure.geo_transform, map_pressure.projection,
                      type_map="delta_P").normalize_data()

    data_potential_score = (map_delta_P.data + norm_residual_recoverable_reserves.data
                            + norm_last_rate_oil.data + norm_init_rate_oil.data) / 4
    map_potential_score = Map(data_potential_score, map_pressure.geo_transform, map_pressure.projection,
                              "potential_score")

    map_potential_score.save_img(f"{save_directory}/map_potential_score.png", data_wells)

    return map_potential_score


def risk_score(map_water_cut, map_initial_oil_saturation) -> Map:
    """
    Оценка проблем
    -------
    Map(type_map=risk_score)
    """
    data_last_oil_saturation = 1 - map_water_cut.data / 100
    mask = np.isnan(map_water_cut.data)
    data_oil_saturation = np.where(mask, map_initial_oil_saturation.data, data_last_oil_saturation)

    # Применение гауссова фильтра для сглаживания при объединении карт обводненности и начальной нефтенасыщенности
    sigma = 5  # параметр для определения степени сглаживания
    data_oil_saturation = gaussian_filter(data_oil_saturation, sigma=sigma)

    data_risk_score = data_oil_saturation
    map_risk_score = Map(data_risk_score, map_water_cut.geo_transform, map_water_cut.projection,
                         "risk_score")

    map_risk_score.save_img(f"{save_directory}/map_risk_score.png", data_wells)
    return map_risk_score


def opportunity_index(map_reservoir_score, map_potential_score, map_risk_score) -> Map:
    """
    Оценка индекса возможностей
    -------
    Map(type_map=opportunity_index)
    """
    k_reservoir = k_potential = k_risk = 1
    data_opportunity_index = (k_reservoir * map_reservoir_score.data +
                              k_potential * map_potential_score.data +
                              k_risk * map_risk_score.data) / 3
    map_opportunity_index = Map(data_opportunity_index, map_reservoir_score.geo_transform,
                                map_reservoir_score.projection, "opportunity_index")
    return map_opportunity_index


def clusterization_zones(map_opportunity_index, epsilon=15, min_samples=50, percent_low=60):
    """Кластеризация зон бурения на основе карты индекса возможности с помощью метода DBSCAN"""
    data_opportunity_index = map_opportunity_index.data

    # Фильтрация карты индекса вероятности по процентилю
    nan_opportunity_index = np.where(data_opportunity_index == 0, np.nan, data_opportunity_index)
    threshold_value = np.nanpercentile(nan_opportunity_index, percent_low)
    data_opportunity_index_threshold = np.where(data_opportunity_index > threshold_value, data_opportunity_index, 0)

    map_opportunity_index_threshold = Map(data_opportunity_index_threshold, map_opportunity_index.geo_transform,
                                          map_opportunity_index.projection, "opportunity_index_threshold")
    map_opportunity_index_threshold.save_img(f"{save_directory}/map_opportunity_index_threshold.png", data_wells)

    # Массив для кластеризации
    drilling_index_map = data_opportunity_index_threshold.copy()

    # Преобразование карты в двумерный массив координат, значений и вытягивание их в вектор
    X, Y = np.meshgrid(np.arange(drilling_index_map.shape[1]), np.arange(drilling_index_map.shape[0]))
    X, Y, Z = X.flatten(), Y.flatten(), drilling_index_map.flatten()

    # Фильтрация точек для обучающего набора данных - на fit идут точки с не нулевым индексом
    X, Y, Z = np.array(X[Z > 0]), np.array(Y[Z > 0]), np.array(Z[Z > 0])

    # Комбинирование координат и значений в один массив
    dataset = pd.DataFrame(np.column_stack((X, Y, Z)), columns=["X", "Y", "Z"])
    training_dataset = dataset[["X", "Y"]]

    # eps: Максимальное расстояние между двумя точками, чтобы одна была соседкой другой
    # min_samples: Минимальное количество точек для образования плотной области
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(training_dataset)
    labels = sorted(list(set(dbscan.labels_)))

    # Создание параметра по которому будет произведена сортировка кластеров // среднее значение индекса в кластере
    mean_indexes = list(map(lambda label: np.mean(Z[np.where(dbscan.labels_ == label)]), labels[1:]))
    mean_indexes = pd.DataFrame({"labels": labels[1:], "mean_indexes": mean_indexes})
    mean_indexes = mean_indexes.sort_values(by=['mean_indexes'], ascending=False).reset_index(drop=True)

    # Создание словаря с зонами
    dict_zones = {}
    for label in labels:
        idx = np.where(dbscan.labels_ == label)
        x_label, y_label, z_label = X[idx], Y[idx], Z[idx]
        mean_index, max_index, position = 0, 0, -1
        if label != -1:
            # Расчет среднего и максимального индекса в кластере
            mean_index = np.float64(np.round(np.mean(z_label), 3))
            max_index = np.float64(np.round(np.max(z_label), 3))
            position = mean_indexes[mean_indexes["labels"] == label].index[0]
        dict_zones[position] = [x_label, y_label, z_label, mean_index, max_index]
    # Добавление в словарь гиперпараметров кластеризации
    dict_zones["DBSCAN_parameters"] = [epsilon, min_samples]
    return dict_zones


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
    # import datetime
    # from dateutil.relativedelta import relativedelta
    # Обработка массива скважин // выбор действующих скважин на последние NUMBER_MONTHS
    # NUMBER_MONTHS = 12
    # last_date = data_wells.date.sort_values()
    # last_date = last_date.unique()[-1]
    # active_date = last_date - relativedelta(months=NUMBER_MONTHS)
    # data_wells_with_work = data_wells[(data_wells.Ql_rate > 0) | (data_wells.Winj_rate > 0)]
    # data_active_wells = data_wells_with_work[data_wells_with_work.date > active_date]
    data_active_wells = data_wells

    logger.info("Расчет буфера вокруг скважин")
    union_buffer = active_well_outline(data_active_wells, buffer_radius=200)

    logger.info("Создание маски буфера")
    if union_buffer:
        mask = create_mask_from_buffers(base_map, union_buffer)
    else:
        mask = np.empty(base_map.data.shape)

    logger.info("Бланкование карты согласно буферу")
    modified_map = cut_map_by_mask(base_map, mask, blank_value=0)

    return modified_map


def active_well_outline(df_wells, buffer_radius=500):
    """
    Создание буфера вокруг действующих скважин
     Parameters
    ----------
    df_wells - DataFrame скважин с обязательными столбцами:
                [well type, T1_x, T1_y, T3_x,T3_y]
    buffer_radius - расстояние от скважин, на котором нельзя бурить // в перспективе замена на радиус дренирования,
     нагнетания с индивидуальным расчетом для каждой скважины

    Returns
    -------
    union_buffer - POLYGON
    """

    if df_wells.empty:
        logger.warning('Нет скважин для создания маски')
        return

    # Создание геометрии для каждой скважины
    def create_buffer(row):
        if row["well type"] == "horizontal":
            # Горизонтальная скважина (линия)
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(row.T1_x, row.T1_y)
            line.AddPoint(row.T3_x, row.T3_y)
            buffer = line.Buffer(buffer_radius)
        else:
            # Вертикальная скважина (точка)
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(row.T1_x, row.T1_y)
            buffer = point.Buffer(buffer_radius)
        return buffer

    # Создаём список буферов
    buffers = [create_buffer(row) for _, row in df_wells.iterrows()]

    # Создаем объединенный буфер
    union_buffer = ogr.Geometry(ogr.wkbGeometryCollection)
    for buffer in buffers:
        union_buffer = union_buffer.Union(buffer)

    return union_buffer


def create_mask_from_buffers(base_map, buffer):
    """
    Функция создания маски из буфера в видe array
    Parameters
    ----------
    map - карта, с которой получаем geo_transform для определения сетки для карты с буфером
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


"""________БЛОК ДЛЯ УДАЛЕНИЯ_______"""
maps_directory = paths["maps_directory"]
data_well_directory = paths["data_well_directory"]
save_directory = paths["save_directory"]
_, data_wells = load_wells_data(data_well_directory=data_well_directory)
"""________БЛОК ДЛЯ УДАЛЕНИЯ_______"""

if __name__ == '__main__':
    # Скрипт для перебора гиперпараметров DBSCAN по карте cut_map_opportunity_index.grd

    from map import read_raster

    map_opportunity_index = read_raster(f"{save_directory}/cut_map_opportunity_index.grd", no_value=0)

    # Перебор параметров DBSCAN c сеткой графиков 5 х 3
    pairs_of_hyperparams = [[5, 20], [5, 50], [5, 100],
                            [10, 20], [10, 50], [10, 100],
                            [15, 20], [15, 50], [15, 100],
                            [20, 20], [20, 50], [20, 100],
                            [30, 20], [30, 50], [30, 100], ]
    fig = plt.figure()
    fig.set_size_inches(20, 50)

    for i, s in enumerate(pairs_of_hyperparams):
        dict_zones = clusterization_zones(map_opportunity_index, epsilon=s[0], min_samples=s[1], percent_low=80)

        ax_ = fig.add_subplot(5, 3, i + 1)

        # Определение размера осей
        x = (map_opportunity_index.geo_transform[0], map_opportunity_index.geo_transform[0] +
             map_opportunity_index.geo_transform[1] * map_opportunity_index.data.shape[1])
        y = (map_opportunity_index.geo_transform[3] + map_opportunity_index.geo_transform[5] *
             map_opportunity_index.data.shape[0], map_opportunity_index.geo_transform[3])

        d_x = x[1] - x[0]
        d_y = y[1] - y[0]

        element_size = min(d_x, d_y) / 10 ** 5
        font_size = min(d_x, d_y) / 10 ** 3

        plt.imshow(map_opportunity_index.data, cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=font_size)

        if data_wells is not None:
            # Преобразование координат скважин в пиксельные координаты
            x_t1, y_t1 = map_opportunity_index.convert_coord((data_wells.T1_x, data_wells.T1_y))
            x_t3, y_t3 = map_opportunity_index.convert_coord((data_wells.T3_x, data_wells.T3_y))

            # Отображение скважин на карте
            plt.plot([x_t1, x_t3], [y_t1, y_t3], c='black', linewidth=element_size)
            plt.scatter(x_t1, y_t1, s=element_size, c='black', marker="o")

            # Отображение имен скважин рядом с точками T1
            for x, y, name in zip(x_t1, y_t1, data_wells.well_number):
                plt.text(x + 3, y - 3, name, fontsize=font_size / 10, ha='left')

        plt.title(map_opportunity_index.type_map, fontsize=font_size * 1.2)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.contour(map_opportunity_index.data, levels=8, colors='black', origin='lower', linewidths=font_size / 100)

        labels = list(set(dict_zones.keys()) - {"DBSCAN_parameters"})
        # Выбираем теплую цветовую карту
        cmap = plt.get_cmap('Wistia', len(set(labels)))
        # Генерируем список цветов
        colors = [cmap(i) for i in range(len(set(labels)))]

        if len(labels) == 1 and labels[0] == -1:
            colors = {0: "gray"}
        else:
            colors = dict(zip(labels, colors))
            colors.update({-1: "gray"})

        for lab, c in zip(labels, colors.values()):
            x_zone = dict_zones[lab][0]
            y_zone = dict_zones[lab][1]
            mean_index = dict_zones[lab][3]
            max_index = dict_zones[lab][4]
            plt.scatter(x_zone, y_zone, color=c, alpha=0.6, s=1)

            x_middle = x_zone[int(len(x_zone) / 2)]
            y_middle = y_zone[int(len(y_zone) / 2)]

            if lab != -1:
                # Отображение среднего и максимального индексов рядом с кластерами
                plt.text(x_middle, y_middle,f"OI_mean = {np.round(mean_index, 2)}",
                         fontsize=font_size / 10, ha='left', color="black")
                plt.text(x_middle, y_middle,f"OI_max = {np.round(max_index, 2)}",
                         fontsize=font_size / 10, ha='left', color="black")

        plt.xlim(0, map_opportunity_index.data.shape[1])
        plt.ylim(0, map_opportunity_index.data.shape[0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().invert_yaxis()

        n_clusters = len(labels) - 1

        plt.title(f"Epsilon = {s[0]}\n min_samples = {s[1]} \n with {n_clusters} clusters")

    fig.tight_layout()
    plt.savefig("D:/Work/Programs_Python/Infill_drilling/output/drilling_index_map", dpi=300)
    plt.close()

    pass
