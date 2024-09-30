from flask import Blueprint, request, jsonify, render_template
from services import *
from metrics import FinancialDataHelper, FinancialMetrics, FinancialAnalysisHelper


views = Blueprint(__name__, "views", template_folder='templates', static_folder='static')


@views.route('/')
def home():
    return render_template("index.html")

@views.route('/trends_documentation')
def google_trends_documentation():
    return render_template("googleTrends.html")

@views.route('/metrics_documentation')
def financial_metrics_documentation():
    return render_template("financialMetrics.html")


# POWERED BY GOOGLE TRENDS
@views.route('/trends/get_google_trends', methods=['GET'])
def get_google_trends():
    """
    Flask API endpoint to fetch Google Trends data, interpret the data, and serve requested sections.
    
    Query parameters:
    - niche (str): Search term for Google Trends.
    - location (str): Location code for the trends (e.g., US, PH).
    - timeframe (str): Time range for trends (default is 1 year).
    - request_type (str): Type of data requested ('trend_summary', 'spike_events', 'breakpoints', 'forecast', 'recommendations').
    
    Returns:
    - JSON: The requested data section in JSON format.
    """
    niche = request.args.get('niche', default='iced coffee', type=str)
    location = request.args.get('location', default='US', type=str)
    timeframe = request.args.get('timeframe', default='today 12-m', type=str)
    request_type = request.args.get('request_type', default='trend_summary', type=str)

    trending_data = get_google_trends_data(niche, timeframe=timeframe, location=location)

    if trending_data is None:
        return jsonify({'error': 'No data found for the given niche or location.'}), 404

    report = interpret_google_trends(trending_data, niche)

    response_json = serve_requested_data(report, request_type)

    return response_json

@views.route('/trends/get_current_month_interest', methods=['GET'])
def get_current_month_interest_api():
    """
    Flask API endpoint to get the interest score for the current month.
    
    Query parameters:
    - niche (str): Search term for Google Trends.
    - location (str): Location code for the trends (e.g., US, PH).
    - timeframe (str): Time range for trends (default is 1 year).
    
    Returns:
    - JSON: Interest score for the current month in JSON format.
    """
    niche = request.args.get('niche', default='iced coffee', type=str)
    location = request.args.get('location', default='US', type=str)
    timeframe = request.args.get('timeframe', default='today 12-m', type=str)

    trending_data = get_google_trends_data(niche, timeframe=timeframe, location=location)

    if trending_data is None:
        return jsonify({'error': 'No data found for the given niche or location.'}), 404

    current_month_interest = get_current_month_interest(trending_data, niche)

    return jsonify(current_month_interest)

@views.route('/trends/compile_all', methods=['GET'])
def compile_all_json_api():
    """
    Flask API endpoint to compile all sections of the report into one JSON response.
    
    Query parameters:
    - niche (str): Search term for Google Trends.
    - location (str): Location code for the trends (e.g., US, PH).
    - timeframe (str): Time range for trends (default is 1 year).
    
    Returns:
    - JSON: All sections of the trends report in JSON format.
    """
    niche = request.args.get('niche', default='iced coffee', type=str)
    location = request.args.get('location', default='US', type=str)
    timeframe = request.args.get('timeframe', default='today 12-m', type=str)

    trending_data = get_google_trends_data(niche, timeframe=timeframe, location=location)

    if trending_data is None:
        return jsonify({'error': 'No data found for the given niche or location.'}), 404

    report = interpret_google_trends(trending_data, niche)

    all_data_json = compile_all_json(report)

    return all_data_json

@views.route('/trends/get-bar-graph-data', methods=['GET'])
def get_bar_graph_data_api():
    niche = request.args.get('niche', 'iced coffee')  # Default to 'iced coffee' if no niche is provided
    timeframe = request.args.get('timeframe', 'today 12-m')
    location = request.args.get('location', 'US')

    # Fetch the Google Trends data
    trending_data = get_google_trends_data(niche, timeframe=timeframe, location=location)

    if trending_data is not None:
        graph_data = get_bar_graph_data(trending_data, niche)
        return graph_data
    else:
        return json.dumps({"error": "No data found for the given niche."}, indent=4)

@views.route('/trends/calculate_roi', methods=['GET'])
def calculate_roi_api():
    """
    Flask API endpoint to calculate ROI based on Google Trends forecast.

    Query parameters:
    - niche (str): Search term for Google Trends.
    - location (str): Location code for the trends (e.g., US, PH).
    - timeframe (str): Time range for trends (default is 1 year).
    - investment_amount (float): Amount of money to invest (default is $1000).
    - forecast_period (int): Days into the future for ROI calculation (default is 90 days).

    Returns:
    - JSON: ROI estimation, forecast data, and recommendations.
    """
    niche = request.args.get('niche', default='iced coffee', type=str)
    location = request.args.get('location', default='US', type=str)
    timeframe = request.args.get('timeframe', default='today 12-m', type=str)
    investment_amount = request.args.get('investment_amount', default=1000, type=float)
    forecast_period = request.args.get('forecast_period', default=90, type=int)

    trending_data = get_google_trends_data(niche, timeframe=timeframe, location=location)

    if trending_data is None:
        return jsonify({'error': 'No data found for the given niche or location.'}), 404

    roi_data = calculate_roi(trending_data, niche, investment_amount, forecast_period)

    return jsonify(roi_data)

@views.route('/trends/business-assessment', methods=['GET'])
def business_assessment():
    current_revenue = request.args.get('current_revenue', type=float)
    previous_revenue = request.args.get('previous_revenue', type=float)
    total_expenses = request.args.get('total_expenses', type=float)
    customer_base = request.args.get('customer_base', type=int)
    months = request.args.get('months', default=12, type=int)
    industry = request.args.get('industry', type=str)

    if None in [current_revenue, previous_revenue, total_expenses, customer_base, months, industry]:
        return jsonify({'error': 'Missing required parameters.'}), 400

    try:
        low_growth_rate_threshold, low_profit_margin_threshold = fetch_industry_benchmarks(industry)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch benchmarks for industry {industry}: {str(e)}'}), 400

    insights = generate_business_insights(
        current_revenue, previous_revenue, total_expenses, customer_base, months,
        low_growth_rate_threshold, low_profit_margin_threshold
    )
    
    return jsonify({
        'insights': insights['suggestions'],
        'growth_rate': insights['growth_rate'],
        'profit_margin': insights['profit_margin'],
        'average_revenue_per_month': insights['average_revenue_per_month'],
        'message': 'Business assessment completed successfully.'
    })
    
@views.route('/trends/interest_by_region', methods=['GET'])
def interest_by_region():
    """
    API endpoint to get Google Trends interest by region for a specific niche.
    """
    niche = request.args.get('niche')
    timeframe = request.args.get('timeframe', 'today 12-m')  # Default to 1 year
    location = request.args.get('location', 'US')  # Default to US

    if not niche:
        return jsonify({'error': 'Niche parameter is required.'}), 400

    # Call the function to get interest by region
    pie_chart_data = get_interest_by_region(niche, timeframe, location)

    # Return the response as JSON
    return jsonify(json.loads(pie_chart_data))  

@views.route('/trends/trend_seeker', methods=['GET'])
def trend_seeker_api():
    # Get parameters from the request
    keyword = request.args.get('niche', default='iced coffee', type=str)
    location = request.args.get('location', default='US', type=str)
    timeframe = request.args.get('timeframe', default='today 12-m', type=str)
    top_n = request.args.get('top_n', default=5, type=int)  # Optional parameter

    # Call the trend_seeker function with the parameters
    result = trend_seeker(keyword, location, timeframe, top_n)

    return jsonify(result)



# FINANCIAL METRICS SECTION
# SECTION 01 - FinancialDataHelper
@views.route('/metrics/calculate_average_total_assets', methods=['GET'])
def calculate_average_total_assets():
    assets = request.args.getlist('assets', type=float)  # Get assets from query parameters
    average_assets = FinancialDataHelper.calculate_average_total_assets(assets)
    return jsonify({'average_total_assets': average_assets})

@views.route('/metrics/calculate_average_inventory', methods=['GET'])
def calculate_average_inventory():
    inventory = request.args.getlist('inventory', type=float)  # Get inventory from query parameters
    average_inventory = FinancialDataHelper.calculate_average_inventory(inventory)
    return jsonify({'average_inventory': average_inventory})

@views.route('/metrics/calculate_average_accounts_receivable', methods=['GET'])
def calculate_average_accounts_receivable():
    accounts_receivable = request.args.getlist('accounts_receivable', type=float)  # Get accounts receivable from query parameters
    average_accounts_receivable = FinancialDataHelper.calculate_average_accounts_receivable(accounts_receivable)
    return jsonify({'average_accounts_receivable': average_accounts_receivable})

@views.route('/metrics/calculate_average_sales', methods=['GET'])
def calculate_average_sales():
    sales = request.args.getlist('sales', type=float)  # Get sales from query parameters
    average_sales = FinancialDataHelper.calculate_average_sales(sales)
    return jsonify({'average_sales': average_sales})

@views.route('/metrics/calculate_cost_of_goods_sold', methods=['GET'])
def calculate_cost_of_goods_sold():
    start_inventory = float(request.args.get('start_inventory', 0))
    purchases = float(request.args.get('purchases', 0))
    end_inventory = float(request.args.get('end_inventory', 0))
    cogs = FinancialDataHelper.calculate_cost_of_goods_sold(start_inventory, purchases, end_inventory)
    return jsonify({'cost_of_goods_sold': cogs})

@views.route('/metrics/get_financial_ratios', methods=['GET'])
def get_financial_ratios():
    total_assets = float(request.args.get('total_assets', 0))
    total_debt = float(request.args.get('total_debt', 0))
    total_equity = float(request.args.get('total_equity', 0))
    financial_ratios = FinancialDataHelper.get_financial_ratios(total_assets, total_debt, total_equity)
    return jsonify(financial_ratios)





# SECTION 02 - FinancialMetrics
@views.route('/metrics/total_revenue', methods=['GET'])
def total_revenue():
    sales = request.args.getlist('sales', type=float)
    result = FinancialMetrics.total_revenue(sales)
    return jsonify({'total_revenue': result})

@views.route('/metrics/revenue_growth_rate', methods=['GET'])
def revenue_growth_rate():
    current_revenue = float(request.args.get('current_revenue'))
    previous_revenue = float(request.args.get('previous_revenue'))
    result = FinancialMetrics.revenue_growth_rate(current_revenue, previous_revenue)
    return jsonify({'revenue_growth_rate': result})

@views.route('/metrics/revenue_per_user', methods=['GET'])
def revenue_per_user():
    total_revenue = float(request.args.get('total_revenue'))
    number_of_users = int(request.args.get('number_of_users'))
    result = FinancialMetrics.revenue_per_user(total_revenue, number_of_users)
    return jsonify({'revenue_per_user': result})

@views.route('/metrics/gross_profit_margin', methods=['GET'])
def gross_profit_margin():
    gross_profit = float(request.args.get('gross_profit'))
    total_revenue = float(request.args.get('total_revenue'))
    result = FinancialMetrics.gross_profit_margin(gross_profit, total_revenue)
    return jsonify({'gross_profit_margin': result})

@views.route('/metrics/operating_profit_margin', methods=['GET'])
def operating_profit_margin():
    operating_income = float(request.args.get('operating_income'))
    total_revenue = float(request.args.get('total_revenue'))
    result = FinancialMetrics.operating_profit_margin(operating_income, total_revenue)
    return jsonify({'operating_profit_margin': result})

@views.route('/metrics/net_profit_margin', methods=['GET'])
def net_profit_margin():
    net_income = float(request.args.get('net_income'))
    total_revenue = float(request.args.get('total_revenue'))
    result = FinancialMetrics.net_profit_margin(net_income, total_revenue)
    return jsonify({'net_profit_margin': result})

@views.route('/metrics/return_on_assets', methods=['GET'])
def return_on_assets():
    net_income = float(request.args.get('net_income'))
    total_assets = float(request.args.get('total_assets'))
    result = FinancialMetrics.return_on_assets(net_income, total_assets)
    return jsonify({'return_on_assets': result})

@views.route('/metrics/return_on_equity', methods=['GET'])
def return_on_equity():
    net_income = float(request.args.get('net_income'))
    shareholder_equity = float(request.args.get('shareholder_equity'))
    result = FinancialMetrics.return_on_equity(net_income, shareholder_equity)
    return jsonify({'return_on_equity': result})

@views.route('/metrics/return_on_investment', methods=['GET'])
def return_on_investment():
    net_profit = float(request.args.get('net_profit'))
    investment_cost = float(request.args.get('investment_cost'))
    result = FinancialMetrics.return_on_investment(net_profit, investment_cost)
    return jsonify({'return_on_investment': result})

@views.route('/metrics/current_ratio', methods=['GET'])
def current_ratio():
    current_assets = float(request.args.get('current_assets'))
    current_liabilities = float(request.args.get('current_liabilities'))
    result = FinancialMetrics.current_ratio(current_assets, current_liabilities)
    return jsonify({'current_ratio': result})

@views.route('/metrics/quick_ratio', methods=['GET'])
def quick_ratio():
    current_assets = float(request.args.get('current_assets'))
    inventories = request.args.getlist('inventories', type=float)
    current_liabilities = float(request.args.get('current_liabilities'))
    result = FinancialMetrics.quick_ratio(current_assets, inventories, current_liabilities)
    return jsonify({'quick_ratio': result})

@views.route('/metrics/cash_ratio', methods=['GET'])
def cash_ratio():
    cash_and_equivalents = float(request.args.get('cash_and_equivalents'))
    current_liabilities = float(request.args.get('current_liabilities'))
    result = FinancialMetrics.cash_ratio(cash_and_equivalents, current_liabilities)
    return jsonify({'cash_ratio': result})

@views.route('/metrics/asset_turnover_ratio', methods=['GET'])
def asset_turnover_ratio():
    total_revenue = float(request.args.get('total_revenue'))
    average_total_assets = float(request.args.get('average_total_assets'))
    result = FinancialMetrics.asset_turnover_ratio(total_revenue, average_total_assets)
    return jsonify({'asset_turnover_ratio': result})

@views.route('/metrics/inventory_turnover_ratio', methods=['GET'])
def inventory_turnover_ratio():
    cost_of_goods_sold = float(request.args.get('cost_of_goods_sold'))
    average_inventory = float(request.args.get('average_inventory'))
    result = FinancialMetrics.inventory_turnover_ratio(cost_of_goods_sold, average_inventory)
    return jsonify({'inventory_turnover_ratio': result})

@views.route('/metrics/receivables_turnover_ratio', methods=['GET'])
def receivables_turnover_ratio():
    net_credit_sales = float(request.args.get('net_credit_sales'))
    average_accounts_receivable = float(request.args.get('average_accounts_receivable'))
    result = FinancialMetrics.receivables_turnover_ratio(net_credit_sales, average_accounts_receivable)
    return jsonify({'receivables_turnover_ratio': result})

@views.route('/metrics/price_to_earnings', methods=['GET'])
def price_to_earnings():
    market_price_per_share = float(request.args.get('market_price_per_share'))
    earnings_per_share = float(request.args.get('earnings_per_share'))
    result = FinancialMetrics.price_to_earnings(market_price_per_share, earnings_per_share)
    return jsonify({'price_to_earnings': result})

@views.route('/metrics/price_to_sales', methods=['GET'])
def price_to_sales():
    market_capitalization = float(request.args.get('market_capitalization'))
    total_revenue = float(request.args.get('total_revenue'))
    result = FinancialMetrics.price_to_sales(market_capitalization, total_revenue)
    return jsonify({'price_to_sales': result})

@views.route('/metrics/enterprise_value', methods=['GET'])
def enterprise_value():
    market_capitalization = float(request.args.get('market_capitalization'))
    total_debt = float(request.args.get('total_debt'))
    cash_and_equivalents = float(request.args.get('cash_and_equivalents'))
    result = FinancialMetrics.enterprise_value(market_capitalization, total_debt, cash_and_equivalents)
    return jsonify({'enterprise_value': result})

@views.route('/metrics/ev_to_ebitda', methods=['GET'])
def ev_to_ebitda():
    enterprise_value = float(request.args.get('enterprise_value'))
    ebitda = float(request.args.get('ebitda'))
    result = FinancialMetrics.ev_to_ebitda(enterprise_value, ebitda)
    return jsonify({'ev_to_ebitda': result})

@views.route('/metrics/earnings_per_share', methods=['GET'])
def earnings_per_share():
    net_income = float(request.args.get('net_income'))
    number_of_outstanding_shares = int(request.args.get('number_of_outstanding_shares'))
    result = FinancialMetrics.earnings_per_share(net_income, number_of_outstanding_shares)
    return jsonify({'earnings_per_share': result})

@views.route('/metrics/eps_growth_rate', methods=['GET'])
def eps_growth_rate():
    current_eps = float(request.args.get('current_eps'))
    previous_eps = float(request.args.get('previous_eps'))
    result = FinancialMetrics.eps_growth_rate(current_eps, previous_eps)
    return jsonify({'eps_growth_rate': result})

@views.route('/metrics/free_cash_flow', methods=['GET'])
def free_cash_flow():
    cash_from_operations = float(request.args.get('cash_from_operations'))
    capital_expenditures = float(request.args.get('capital_expenditures'))
    result = FinancialMetrics.free_cash_flow(cash_from_operations, capital_expenditures)
    return jsonify({'free_cash_flow': result})

@views.route('/metrics/debt_to_equity', methods=['GET'])
def debt_to_equity():
    total_debt = float(request.args.get('total_debt'))
    total_equity = float(request.args.get('total_shareholder_equity'))
    result = FinancialMetrics.debt_to_equity(total_debt, total_equity)
    return jsonify({'debt_to_equity': result})

@views.route('/metrics/debt_ratio', methods=['GET'])
def debt_ratio():
    total_debt = float(request.args.get('total_debt'))
    total_assets = float(request.args.get('total_assets'))
    result = FinancialMetrics.debt_ratio(total_debt, total_assets)
    return jsonify({'debt_ratio': result})

@views.route('/metrics/interest_coverage_ratio', methods=['GET'])
def interest_coverage_ratio():
    ebit = float(request.args.get('ebit'))
    interest_expense = float(request.args.get('interest_expense'))
    result = FinancialMetrics.interest_coverage_ratio(ebit, interest_expense)
    return jsonify({'interest_coverage_ratio': result})

@views.route('/metrics/market_share', methods=['GET'])
def market_share():
    company_sales = float(request.args.get('company_sales'))
    total_market_sales = float(request.args.get('total_market_sales'))
    result = FinancialMetrics.market_share(company_sales, total_market_sales)
    return jsonify({'market_share': result})

@views.route('/metrics/relative_market_share', methods=['GET'])
def relative_market_share():
    company_market_share = float(request.args.get('company_market_share'))
    competitor_market_share = float(request.args.get('competitor_market_share'))
    result = FinancialMetrics.relative_market_share(company_market_share, competitor_market_share)
    return jsonify({'relative_market_share': result})

@views.route('/metrics/customer_acquisition_cost', methods=['GET'])
def customer_acquisition_cost():
    total_sales_and_marketing_expenses = float(request.args.get('total_sales_and_marketing_expenses'))
    number_of_new_customers = int(request.args.get('number_of_new_customers'))
    result = FinancialMetrics.customer_acquisition_cost(total_sales_and_marketing_expenses, number_of_new_customers)
    return jsonify({'customer_acquisition_cost': result})

@views.route('/metrics/customer_lifetime_value', methods=['GET'])
def customer_lifetime_value():
    average_purchase_value = float(request.args.get('average_purchase_value'))
    purchase_frequency = float(request.args.get('purchase_frequency'))
    customer_lifespan = float(request.args.get('customer_lifespan'))
    result = FinancialMetrics.customer_lifetime_value(average_purchase_value, purchase_frequency, customer_lifespan)
    return jsonify({'customer_lifetime_value': result})





# SECTION 03 - FinancialAnalysisHelper
@views.route('/metrics/financial_analysis', methods=['GET'])
def financial_analysis():
    assets = list(map(float, request.args.getlist('assets')))
    inventory = list(map(float, request.args.getlist('inventory')))
    accounts_receivable = list(map(float, request.args.getlist('accounts_receivable')))
    sales = list(map(float, request.args.getlist('sales')))
    
    result = FinancialAnalysisHelper.financial_analysis(assets, inventory, accounts_receivable, sales)
    return jsonify(result)

@views.route('/metrics/valuation', methods=['GET'])
def valuation():
    market_price_per_share = float(request.args.get('market_price_per_share'))
    earnings_per_share = float(request.args.get('earnings_per_share'))
    market_capitalization = float(request.args.get('market_capitalization'))
    total_revenue = float(request.args.get('total_revenue'))
    total_debt = float(request.args.get('total_debt'))
    cash_and_equivalents = float(request.args.get('cash_and_equivalents'))
    ebitda = float(request.args.get('ebitda'))
    number_of_outstanding_shares = int(request.args.get('number_of_outstanding_shares'))
    
    result = FinancialAnalysisHelper.valuation(market_price_per_share, earnings_per_share, market_capitalization, total_revenue, total_debt, cash_and_equivalents, ebitda, number_of_outstanding_shares)
    return jsonify(result)

@views.route('/metrics/performance_measurement', methods=['GET'])
def performance_measurement():
    net_income = float(request.args.get('net_income'))
    total_assets = float(request.args.get('total_assets'))
    total_equity = float(request.args.get('total_equity'))
    shareholder_equity = float(request.args.get('shareholder_equity'))
    investment_cost = float(request.args.get('investment_cost'))
    current_assets = float(request.args.get('current_assets'))
    current_liabilities = float(request.args.get('current_liabilities'))
    inventory = list(map(float, request.args.getlist('inventory')))
    
    result = FinancialAnalysisHelper.performance_measurement(net_income, total_assets, total_equity, shareholder_equity, investment_cost, current_assets, current_liabilities, inventory)
    return jsonify(result)

@views.route('/metrics/risk_analysis', methods=['GET'])
def risk_analysis():
    portfolio_return = float(request.args.get('portfolio_return'))
    risk_free_rate = float(request.args.get('risk_free_rate'))
    portfolio_std_dev = float(request.args.get('portfolio_std_dev'))
    benchmark_return = float(request.args.get('benchmark_return'))
    beta = float(request.args.get('beta'))
    
    result = FinancialAnalysisHelper.risk_analysis(portfolio_return, risk_free_rate, portfolio_std_dev, benchmark_return, beta)
    return jsonify(result)

@views.route('/metrics/market_share_analysis', methods=['GET'])
def market_share_analysis():
    company_sales = float(request.args.get('company_sales'))
    total_market_sales = float(request.args.get('total_market_sales'))
    competitor_market_share = float(request.args.get('competitor_market_share'))
    
    result = FinancialAnalysisHelper.market_share_analysis(company_sales, total_market_sales, competitor_market_share)
    return jsonify(result)

@views.route('/metrics/business_health_analysis', methods=['GET'])
def business_health_analysis():
    assets = list(map(float, request.args.getlist('assets')))
    inventory = list(map(float, request.args.getlist('inventory')))
    accounts_receivable = list(map(float, request.args.getlist('accounts_receivable')))
    sales = list(map(float, request.args.getlist('sales')))
    market_price_per_share = float(request.args.get('market_price_per_share'))
    earnings_per_share = float(request.args.get('earnings_per_share'))
    market_capitalization = float(request.args.get('market_capitalization'))
    total_revenue = float(request.args.get('total_revenue'))
    total_debt = float(request.args.get('total_debt'))
    cash_and_equivalents = float(request.args.get('cash_and_equivalents'))
    ebitda = float(request.args.get('ebitda'))
    number_of_outstanding_shares = int(request.args.get('number_of_outstanding_shares'))
    net_income = float(request.args.get('net_income'))
    total_assets = float(request.args.get('total_assets'))
    total_equity = float(request.args.get('total_equity'))
    shareholder_equity = float(request.args.get('shareholder_equity'))
    investment_cost = float(request.args.get('investment_cost'))
    current_assets = float(request.args.get('current_assets'))
    current_liabilities = float(request.args.get('current_liabilities'))
    competitor_market_share = float(request.args.get('competitor_market_share'))
    portfolio_return = float(request.args.get('portfolio_return'))
    risk_free_rate = float(request.args.get('risk_free_rate'))
    portfolio_std_dev = float(request.args.get('portfolio_std_dev'))
    benchmark_return = float(request.args.get('benchmark_return'))
    beta = float(request.args.get('beta'))

    result = FinancialAnalysisHelper.business_health_analysis(
        assets, inventory, accounts_receivable, sales, 
        market_price_per_share, earnings_per_share, 
        market_capitalization, total_revenue, total_debt, 
        cash_and_equivalents, ebitda, number_of_outstanding_shares, 
        net_income, total_assets, total_equity, shareholder_equity, 
        investment_cost, current_assets, current_liabilities, 
        competitor_market_share, portfolio_return, risk_free_rate, 
        portfolio_std_dev, benchmark_return, beta)
    
    return jsonify(result)

