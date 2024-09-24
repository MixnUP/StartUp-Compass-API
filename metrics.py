class FinancialDataHelper:
    @staticmethod
    def calculate_average_total_assets(assets):
        """Calculate the average total assets from a list of asset values."""
        return sum(assets) / len(assets) if assets else 0

    @staticmethod
    def calculate_average_inventory(inventory):
        """Calculate the average inventory from a list of inventory values."""
        return sum(inventory) / len(inventory) if inventory else 0

    @staticmethod
    def calculate_average_accounts_receivable(accounts_receivable):
        """Calculate the average accounts receivable from a list of values."""
        return sum(accounts_receivable) / len(accounts_receivable) if accounts_receivable else 0

    @staticmethod
    def calculate_average_sales(sales):
        """Calculate the average sales from a list of sales values."""
        return sum(sales) / len(sales) if sales else 0

    @staticmethod
    def calculate_cost_of_goods_sold(start_inventory, purchases, end_inventory):
        """Calculate cost of goods sold (COGS)."""
        return start_inventory + purchases - end_inventory

    @staticmethod
    def get_financial_ratios(total_assets, total_debt, total_equity):
        """Return key financial ratios."""
        return {
            'debt_to_equity': FinancialMetrics.debt_to_equity(total_debt, total_equity),
            'debt_ratio': FinancialMetrics.debt_ratio(total_debt, total_assets),
            'current_ratio': FinancialMetrics.current_ratio(total_assets, total_debt),  # Assuming current liabilities equal total debt for simplicity
        }


class FinancialMetrics:
    @staticmethod
    def total_revenue(sales):
        """Calculate total revenue."""
        return sum(sales)

    @staticmethod
    def revenue_growth_rate(current_revenue, previous_revenue):
        """Calculate revenue growth rate."""
        if previous_revenue == 0:
            return float('inf')  # Avoid division by zero
        return ((current_revenue - previous_revenue) / previous_revenue) * 100

    @staticmethod
    def revenue_per_user(total_revenue, number_of_users):
        """Calculate revenue per user."""
        return total_revenue / number_of_users if number_of_users else 0

    @staticmethod
    def gross_profit_margin(gross_profit, total_revenue):
        """Calculate gross profit margin."""
        return (gross_profit / total_revenue) * 100 if total_revenue else 0

    @staticmethod
    def operating_profit_margin(operating_income, total_revenue):
        """Calculate operating profit margin."""
        return (operating_income / total_revenue) * 100 if total_revenue else 0

    @staticmethod
    def net_profit_margin(net_income, total_revenue):
        """Calculate net profit margin."""
        return (net_income / total_revenue) * 100 if total_revenue else 0

    @staticmethod
    def return_on_assets(net_income, total_assets):
        """Calculate return on assets (ROA)."""
        return (net_income / total_assets) * 100 if total_assets else 0

    @staticmethod
    def return_on_equity(net_income, shareholder_equity):
        """Calculate return on equity (ROE)."""
        return (net_income / shareholder_equity) * 100 if shareholder_equity else 0

    @staticmethod
    def return_on_investment(net_profit, investment_cost):
        """Calculate return on investment (ROI)."""
        return (net_profit / investment_cost) * 100 if investment_cost else 0

    @staticmethod
    def current_ratio(current_assets, current_liabilities):
        """Calculate current ratio."""
        return current_assets / current_liabilities if current_liabilities else float('inf')

    @staticmethod
    def quick_ratio(current_assets, inventories, current_liabilities):
        """Calculate quick ratio."""
        total_inventory = sum(inventories) if inventories else 0
        return (current_assets - total_inventory) / current_liabilities if current_liabilities else float('inf')

    @staticmethod
    def cash_ratio(cash_and_equivalents, current_liabilities):
        """Calculate cash ratio."""
        return cash_and_equivalents / current_liabilities if current_liabilities else float('inf')

    @staticmethod
    def asset_turnover_ratio(total_revenue, average_total_assets):
        """Calculate asset turnover ratio."""
        return total_revenue / average_total_assets if average_total_assets else 0

    @staticmethod
    def inventory_turnover_ratio(cost_of_goods_sold, average_inventory):
        """Calculate inventory turnover ratio."""
        return cost_of_goods_sold / average_inventory if average_inventory else float('inf')

    @staticmethod
    def receivables_turnover_ratio(net_credit_sales, average_accounts_receivable):
        """Calculate receivables turnover ratio."""
        return net_credit_sales / average_accounts_receivable if average_accounts_receivable else float('inf')

    @staticmethod
    def price_to_earnings(market_price_per_share, earnings_per_share):
        """Calculate price-to-earnings (P/E) ratio."""
        return market_price_per_share / earnings_per_share if earnings_per_share else float('inf')

    @staticmethod
    def price_to_sales(market_capitalization, total_revenue):
        """Calculate price-to-sales (P/S) ratio."""
        return market_capitalization / total_revenue if total_revenue else float('inf')

    @staticmethod
    def enterprise_value(market_capitalization, total_debt, cash_and_equivalents):
        """Calculate enterprise value (EV)."""
        return market_capitalization + total_debt - cash_and_equivalents

    @staticmethod
    def ev_to_ebitda(enterprise_value, ebitda):
        """Calculate EV/EBITDA."""
        return enterprise_value / ebitda if ebitda else float('inf')

    @staticmethod
    def earnings_per_share(net_income, number_of_outstanding_shares):
        """Calculate earnings per share (EPS)."""
        return net_income / number_of_outstanding_shares if number_of_outstanding_shares else 0

    @staticmethod
    def eps_growth_rate(current_eps, previous_eps):
        """Calculate EPS growth rate."""
        return ((current_eps - previous_eps) / previous_eps) * 100 if previous_eps else float('inf')

    @staticmethod
    def free_cash_flow(cash_from_operations, capital_expenditures):
        """Calculate free cash flow (FCF)."""
        return cash_from_operations - capital_expenditures

    @staticmethod
    def debt_to_equity(total_debt, total_equity):
        """Calculate debt-to-equity ratio."""
        return total_debt / total_equity if total_equity else float('inf')

    @staticmethod
    def debt_ratio(total_debt, total_assets):
        """Calculate debt ratio."""
        return total_debt / total_assets if total_assets else 0

    @staticmethod
    def interest_coverage_ratio(ebit, interest_expense):
        """Calculate interest coverage ratio."""
        return ebit / interest_expense if interest_expense else float('inf')

    @staticmethod
    def market_share(company_sales, total_market_sales):
        """Calculate market share."""
        return company_sales / total_market_sales if total_market_sales else 0

    @staticmethod
    def relative_market_share(company_market_share, competitor_market_share):
        """Calculate relative market share."""
        return company_market_share / competitor_market_share if competitor_market_share else float('inf')

    @staticmethod
    def customer_acquisition_cost(total_sales_and_marketing_expenses, number_of_new_customers):
        """Calculate customer acquisition cost (CAC)."""
        return total_sales_and_marketing_expenses / number_of_new_customers if number_of_new_customers else float('inf')

    @staticmethod
    def customer_lifetime_value(average_purchase_value, purchase_frequency, customer_lifespan):
        """Calculate customer lifetime value (CLV)."""
        return average_purchase_value * purchase_frequency * customer_lifespan

    @staticmethod
    def gdp(gdp_value):
        """Returns the GDP value."""
        return gdp_value

    @staticmethod
    def consumer_price_index(cpi_value):
        """Returns the Consumer Price Index value."""
        return cpi_value

    @staticmethod
    def unemployment_rate(unemployment_value):
        """Returns the unemployment rate."""
        return unemployment_value

    @staticmethod
    def book_value_per_share(total_equity, number_of_outstanding_shares):
        """Calculate book value per share."""
        return total_equity / number_of_outstanding_shares if number_of_outstanding_shares else 0

    @staticmethod
    def dividend_yield(annual_dividends_per_share, market_price_per_share):
        """Calculate dividend yield."""
        return (annual_dividends_per_share / market_price_per_share) * 100 if market_price_per_share else 0

    @staticmethod
    def sharpe_ratio(portfolio_return, risk_free_rate, portfolio_std_dev):
        """Calculate Sharpe ratio."""
        return (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev else float('inf')

    @staticmethod
    def alpha(portfolio_return, benchmark_return, beta):
        """Calculate alpha."""
        return portfolio_return - (benchmark_return * beta)


class FinancialAnalysisHelper:
    @staticmethod
    def financial_analysis(assets, inventory, accounts_receivable, sales):
        """Perform a comprehensive financial analysis."""
        avg_total_assets = FinancialDataHelper.calculate_average_total_assets(assets)
        avg_inventory = FinancialDataHelper.calculate_average_inventory(inventory)
        avg_accounts_receivable = FinancialDataHelper.calculate_average_accounts_receivable(accounts_receivable)
        avg_sales = FinancialDataHelper.calculate_average_sales(sales)

        financial_ratios = FinancialDataHelper.get_financial_ratios(avg_total_assets, sum(assets), sum(inventory))

        return {
            'average_total_assets': avg_total_assets,
            'average_inventory': avg_inventory,
            'average_accounts_receivable': avg_accounts_receivable,
            'average_sales': avg_sales,
            'financial_ratios': financial_ratios
        }

    @staticmethod
    def valuation(market_price_per_share, earnings_per_share, market_capitalization, total_revenue, total_debt, cash_and_equivalents, ebitda, number_of_outstanding_shares):
        """Calculate valuation metrics."""
        pe_ratio = FinancialMetrics.price_to_earnings(market_price_per_share, earnings_per_share)
        ps_ratio = FinancialMetrics.price_to_sales(market_capitalization, total_revenue)
        ev = FinancialMetrics.enterprise_value(market_capitalization, total_debt, cash_and_equivalents)
        ev_ebitda = FinancialMetrics.ev_to_ebitda(ev, ebitda)
        eps = FinancialMetrics.earnings_per_share(total_revenue, number_of_outstanding_shares)

        return {
            'price_to_earnings_ratio': pe_ratio,
            'price_to_sales_ratio': ps_ratio,
            'enterprise_value': ev,
            'EV_to_EBITDA': ev_ebitda,
            'earnings_per_share': eps
        }

    @staticmethod
    def performance_measurement(net_income, total_assets, total_equity, shareholder_equity, investment_cost, current_assets, current_liabilities, inventory):
        """Measure performance indicators."""
        roa = FinancialMetrics.return_on_assets(net_income, total_assets)
        roe = FinancialMetrics.return_on_equity(net_income, shareholder_equity)
        roi = FinancialMetrics.return_on_investment(net_income, investment_cost)
        current_ratio = FinancialMetrics.current_ratio(current_assets, current_liabilities)
        quick_ratio = FinancialMetrics.quick_ratio(current_assets, inventory, current_liabilities)

        return {
            'return_on_assets': roa,
            'return_on_equity': roe,
            'return_on_investment': roi,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio
        }

    @staticmethod
    def risk_analysis(portfolio_return, risk_free_rate, portfolio_std_dev, benchmark_return, beta):
        """Analyze risk with financial data."""
        sharpe_ratio = FinancialMetrics.sharpe_ratio(portfolio_return, risk_free_rate, portfolio_std_dev)
        alpha_value = FinancialMetrics.alpha(portfolio_return, benchmark_return, beta)

        return {
            'sharpe_ratio': sharpe_ratio,
            'alpha': alpha_value
        }

    @staticmethod
    def market_share_analysis(company_sales, total_market_sales, competitor_market_share):
        """Analyze market share metrics."""
        market_share = FinancialMetrics.market_share(company_sales, total_market_sales)
        relative_market_share = FinancialMetrics.relative_market_share(market_share, competitor_market_share)

        return {
            'market_share': market_share,
            'relative_market_share': relative_market_share
        }
        
    # FULL ANALYSIS
    @staticmethod
    def business_health_analysis(assets, inventory, accounts_receivable, sales, market_price_per_share, earnings_per_share, market_capitalization, total_revenue, total_debt, cash_and_equivalents, ebitda, number_of_outstanding_shares, net_income, total_assets, total_equity, shareholder_equity, investment_cost, current_assets, current_liabilities, competitor_market_share, portfolio_return, risk_free_rate, portfolio_std_dev, benchmark_return, beta):
        """Determine the health of a business based on various financial metrics."""
        
        # Financial Analysis
        financial_analysis_result = FinancialAnalysisHelper.financial_analysis(assets, inventory, accounts_receivable, sales)

        # Valuation
        valuation_result = FinancialAnalysisHelper.valuation(market_price_per_share, earnings_per_share, market_capitalization, total_revenue, total_debt, cash_and_equivalents, ebitda, number_of_outstanding_shares)

        # Performance Measurement
        performance_measurement_result = FinancialAnalysisHelper.performance_measurement(net_income, total_assets, total_equity, shareholder_equity, investment_cost, current_assets, current_liabilities, inventory)

        # Market Share Analysis
        market_share_analysis_result = FinancialAnalysisHelper.market_share_analysis(sales[-1], total_revenue, competitor_market_share)

        # Risk Analysis
        risk_analysis_result = FinancialAnalysisHelper.risk_analysis(portfolio_return, risk_free_rate, portfolio_std_dev, benchmark_return, beta)

        # Combine all results for a comprehensive business health analysis
        business_health = {
            'financial_analysis': financial_analysis_result,
            'valuation': valuation_result,
            'performance_measurement': performance_measurement_result,
            'market_share_analysis': market_share_analysis_result,
            'risk_analysis': risk_analysis_result
        }

        return business_health
    