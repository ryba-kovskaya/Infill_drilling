import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
from loguru import logger


def calculate_depreciation_base(capex_series, lifetime):
    """
    Расчет базы для амортизации по методу уменьшаемого остатка.

    capex_series: pd.Series - капитальные затраты (CAPEX) по годам.
    lifetime: int - срок амортизации в годах.

    Возвращает: pd.Series - база для амортизации по годам.
    """
    start_year = capex_series.index.min()
    end_year = capex_series.index.max()

    depreciation_base = pd.Series(0, index=range(start_year, end_year + int(lifetime) + 1))  # Создаем пустую серию

    for year in range(start_year, end_year + int(lifetime) + 1):
        for age in range(int(lifetime) + 1):  # 0 - текущий год, 1 - год назад и т.д.
            contribution_year = year - age
            if contribution_year in capex_series.index:
                if age == lifetime:
                    depreciation_base[year] += 0.5 * capex_series[contribution_year]  # Половина CAPEX за lifetime назад
                elif age == 0:
                    depreciation_base[year] += 0.5 * capex_series[year]  # Половина текущего CAPEX
                else:
                    depreciation_base[year] += capex_series[contribution_year]  # Полный CAPEX за прошлые годы

    return depreciation_base  # Ограничиваем диапазон depreciation_base.loc[start_year:end_year + 1]


def linear_depreciation(cost, salvage, life):
    """
    Рассчитывает величину амортизации актива за один период, используя линейный метод.

    :param cost: Начальная стоимость актива.
    :param salvage: Ликвидационная стоимость актива (стоимость в конце срока службы).
    :param life: Срок службы актива (количество периодов).
    :return: Величина амортизации за один период.
    """
    if life <= 0:
        raise ValueError("Срок службы должен быть больше 0")

    return (cost - salvage) / life


def bring_arrays_to_one_date(*series):
    """ Привести датированные показатели к одной дате в массиве"""
    joint_data_frame = series[0].to_frame()
    for part in series[1:]:
        joint_data_frame = joint_data_frame.join(part).fillna(method='ffill')
    return joint_data_frame


def calculate_production_by_years(production_by_month, start_date, type):
    """ Расчет суммарной добычи по годам в тыс. т для Qo и Ql"""
    if type == 'Qo':
        name = 'Qo_yearly'
    elif type == 'Ql':
        name = 'Ql_yearly'
    # Создаем временной ряд с месячным интервалом
    date_range = pd.date_range(start=start_date, periods=len(production_by_month), freq='M')
    # Создаем DataFrame
    series_production_by_month = pd.Series(production_by_month, index=date_range, name=name)
    # Группируем по годам и суммируем добычу
    production_by_years = series_production_by_month.resample('Y').sum() / 1000  # в тыс. т
    # Переименовываем индекс для удобства
    production_by_years.index = production_by_years.index.year
    return production_by_years


def calculate_performance_indicators(income, OPEX, CAPEX, df_taxes, df_depreciation, penalty_gas_flaring):
    """ Расчет показателей эффективности: EBITDA, EBIT, NOPAT, OCF, FCF, Накопленный поток наличности"""
    df_indicators = bring_arrays_to_one_date(income, OPEX, CAPEX, df_taxes, df_depreciation, penalty_gas_flaring)
    # EBITDA = Выручка - OPEX - НДПИ нефть - Налог на имущество - НДД
    df_indicators['EBITDA'] = (df_indicators.income - df_indicators.OPEX
                               - df_indicators.taxes + df_indicators.profits_tax - penalty_gas_flaring)
    # EBIT = EBITDA - Амортизация для Налога на прибыль (с учетом премии 30%)
    df_indicators['EBIT'] = df_indicators.EBITDA - df_indicators.depreciation_income_tax
    # NOPAT = EBIT - Налог на прибыль
    df_indicators['NOPAT'] = df_indicators.EBIT - df_indicators.profits_tax
    # OCF = NOPAT + Амортизация для Налога на прибыль (с учетом премии 30%)
    df_indicators['OCF'] = df_indicators.NOPAT + df_indicators.depreciation_income_tax
    # ICF = -CAPEX
    df_indicators['ICF'] = -df_indicators.CAPEX
    # FCF = OCF+ICF
    df_indicators['FCF'] = df_indicators.OCF + df_indicators.ICF
    # Накопленный поток наличности
    df_indicators['cumulative_cash_flow'] = df_indicators.FCF.cumsum()
    return df_indicators[['EBITDA', 'EBIT', 'NOPAT', 'OCF', 'FCF', 'cumulative_cash_flow']]
    # return df_indicators


def calculate_irr_root_scalar(cashflows):
    """Расчет IRR с помощью scipy.optimize.root_scalar"""
    def npv(rate):
        return np.sum([cf / (1 + rate) ** i for i, cf in enumerate(cashflows)])
    try:
        result = root_scalar(npv, bracket=[-1.5, 10], method='brentq')  # Метод Брента
        return result.root if result.converged else None
    except ValueError:
        return None


def calculate_mirr(cashflows, finance_rate=0, reinvest_rate=0):
    """
    Рассчитывает модифицированную внутреннюю норму доходности (MIRR).

    cashflows: array-like — денежные потоки по годам.
    finance_rate: float — ставка дисконтирования для затрат (стоимость капитала).
    reinvest_rate: float — ставка реинвестирования положительных потоков.
    """
    years = len(cashflows) - 1
    negatives = [cf / (1 + finance_rate) ** i for i, cf in enumerate(cashflows) if cf < 0]
    positives = [cf * (1 + reinvest_rate) ** (years - i) for i, cf in enumerate(cashflows) if cf > 0]

    pv_negatives = abs(sum(negatives))
    fv_positives = sum(positives)

    return (fv_positives / pv_negatives) ** (1 / years) - 1 if pv_negatives > 0 else None


def calculation_Kg(Kg_group, production_reserves):
    """Расчет коэффициента Kг в схеме НДД"""
    year_commercial_development = (production_reserves > 0.01).cumsum()
    if Kg_group == 1 or Kg_group == 2:
        return year_commercial_development.apply(lambda x: 0.4 if x < 7 else 0.6 if x == 7 else 0.8 if x == 8 else 1)
    elif Kg_group == 3:
        return year_commercial_development.apply(lambda x: 1)
    elif Kg_group == 4:
        return year_commercial_development.apply(lambda x: 0.5 if x < 3 else 0.75 if x < 4 else 1)
    else:
        logger.warning(f"Неверное значение Кг_номер группы: {Kg_group}")
