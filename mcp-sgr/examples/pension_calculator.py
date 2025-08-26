"""Pension savings calculator with inflation and optional indexation.

Inputs:
- current age
- desired retirement age
- desired monthly pension (in today's money)
- average annual return rate (nominal)
- annual inflation rate
- retirement duration in years (default: 25)
- whether to index pension by inflation during retirement

Outputs:
- Required monthly contribution during accumulation period
- Target capital at retirement
- Key assumptions summary
"""

from __future__ import annotations

import argparse
from math import isclose


def monthly_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def required_retirement_capital(
    first_month_pension_nominal: float,
    monthly_return_rate: float,
    monthly_inflation_rate: float,
    retirement_months: int,
    indexed: bool,
) -> float:
    if retirement_months <= 0:
        return 0.0

    i = monthly_return_rate
    g = monthly_inflation_rate if indexed else 0.0

    # If return ~ inflation (or worse), fall back to simple sum of payments
    if i <= g or isclose(i, g, rel_tol=1e-9, abs_tol=1e-12):
        return first_month_pension_nominal * retirement_months

    # Present value of a growing annuity (payments at end of month)
    # PV = P * (1 - ((1+g)/(1+i))^N) / (i - g)
    ratio = (1.0 + g) / (1.0 + i)
    pv = first_month_pension_nominal * (1.0 - (ratio**retirement_months)) / (i - g)
    return max(0.0, pv)


def required_monthly_contribution(
    target_fv: float,
    monthly_return_rate: float,
    accumulation_months: int,
) -> float:
    if accumulation_months <= 0:
        return 0.0

    i = monthly_return_rate
    if isclose(i, 0.0, abs_tol=1e-12):
        # No growth: simple accumulation
        return target_fv / float(accumulation_months)

    # Future value of annuity due at end of month: FV = c * ((1+i)^N - 1) / i
    denom = (1.0 + i) ** accumulation_months - 1.0
    if denom <= 0:
        return target_fv / float(accumulation_months)
    c = target_fv * i / denom
    return max(0.0, c)


def compute(
    current_age: int,
    retire_age: int,
    desired_monthly_pension_today: float,
    annual_return_rate: float,
    annual_inflation_rate: float,
    retirement_years: int = 25,
    indexation: bool = True,
) -> dict:
    years_to_retire = max(0, retire_age - current_age)
    months_to_retire = years_to_retire * 12
    retirement_months = max(0, retirement_years * 12)

    i_m = monthly_rate(annual_return_rate)
    g_m = monthly_rate(annual_inflation_rate)

    # Convert desired pension to nominal at retirement (first month)
    first_month_pension_nominal = desired_monthly_pension_today * (1.0 + annual_inflation_rate) ** (
        years_to_retire
    )

    target_capital = required_retirement_capital(
        first_month_pension_nominal=first_month_pension_nominal,
        monthly_return_rate=i_m,
        monthly_inflation_rate=g_m,
        retirement_months=retirement_months,
        indexed=indexation,
    )

    monthly_contrib = required_monthly_contribution(
        target_fv=target_capital, monthly_return_rate=i_m, accumulation_months=months_to_retire
    )

    return {
        "monthly_contribution": monthly_contrib,
        "target_capital_at_retirement": target_capital,
        "assumptions": {
            "years_to_retirement": years_to_retire,
            "retirement_years": retirement_years,
            "annual_return_rate": annual_return_rate,
            "annual_inflation_rate": annual_inflation_rate,
            "monthly_return_rate": i_m,
            "monthly_inflation_rate": g_m,
            "indexation": indexation,
            "first_month_pension_nominal": first_month_pension_nominal,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Pension savings calculator with inflation/indexation")
    p.add_argument("--current-age", type=int, required=True)
    p.add_argument("--retire-age", type=int, required=True)
    p.add_argument("--monthly-pension", type=float, required=True, help="Desired monthly pension in today's money")
    p.add_argument("--annual-return", type=float, required=True, help="Annual nominal return, e.g. 0.08 for 8%")
    p.add_argument("--annual-inflation", type=float, required=True, help="Annual inflation, e.g. 0.05 for 5%")
    p.add_argument("--retirement-years", type=int, default=25)
    p.add_argument("--no-indexation", action="store_true", help="Disable pension indexation during retirement")

    args = p.parse_args()
    res = compute(
        current_age=args.current_age,
        retire_age=args.retire_age,
        desired_monthly_pension_today=args.monthly_pension,
        annual_return_rate=args.annual_return,
        annual_inflation_rate=args.annual_inflation,
        retirement_years=args.retirement_years,
        indexation=(not args.no_indexation),
    )

    # Pretty print
    def money(x: float) -> str:
        return f"{x:,.2f}".replace(",", " ")

    print("Pension calculator results")
    print("-" * 32)
    print(f"Required monthly contribution: {money(res['monthly_contribution'])}")
    print(f"Target capital at retirement:  {money(res['target_capital_at_retirement'])}")
    print("Assumptions:")
    a = res["assumptions"]
    print(f"  Years to retirement: {a['years_to_retirement']}")
    print(f"  Retirement years:    {a['retirement_years']}")
    print(f"  Annual return:       {a['annual_return_rate']*100:.2f}%")
    print(f"  Annual inflation:    {a['annual_inflation_rate']*100:.2f}%")
    print(f"  Indexation:          {a['indexation']}")
    print(f"  First pension (nominal at retirement): {money(a['first_month_pension_nominal'])}")


if __name__ == "__main__":
    main()

