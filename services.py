from pytrends.request import TrendReq
import pandas as pd
import json
import numpy as np
from prophet import Prophet
from ruptures import Pelt
from datetime import datetime
import yfinance as yf
import time
# with plotly


def get_current_month_interest(df, niche):
    """
    Retrieve the interest score for the current month based on the provided Google Trends data.

    Parameters:
    - df (pd.DataFrame): Google Trends data.
    - niche (str): Search term.

    Returns:
    - dict: Dictionary containing the interest score for the current month.
    """
    # Get the current month and year
    current_month = datetime.now().month
    current_year = datetime.now().year

    # Filter the dataframe for the current month and year
    df['Date'] = pd.to_datetime(df.index)
    current_month_data = df[(df['Date'].dt.month == current_month) & (df['Date'].dt.year == current_year)]

    # Check if there is data for the current month
    if current_month_data.empty:
        return {'error': 'No data available for the current month.'}

    # Get the latest interest score or the average for the current month
    latest_interest_score = int(current_month_data[niche].iloc[-1])  # Convert to standard Python int
    average_interest_score = float(current_month_data[niche].mean())  # Convert to standard Python float

    return {
        'latest_interest_score': latest_interest_score,
        'average_interest_score': average_interest_score,
        'month': current_month,
        'year': current_year
    }

def get_google_trends_data(niche, timeframe='today 12-m', location='US'):
    """
    Fetch Google Trends data for a niche with adjustable timeframe and location.
    
    Parameters:
    - niche (str): Search term for Google Trends.
    - timeframe (str): Time range for trends (default is 1 year).
    - location (str): Location (geo) for the trends data.
    
    Returns:
    - pd.DataFrame: Google Trends data for the niche.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([niche], timeframe=timeframe, geo=location, gprop='')
    trending_data = pytrends.interest_over_time()

    # Check if data exists
    if trending_data.empty:
        print("No data found for the given niche.")
        return None

    trending_data = trending_data.drop(columns=['isPartial'], errors='ignore')  # Remove 'isPartial' if present
    return trending_data

def generate_recommendations(df, report, niche):
    """
    Generate comprehensive recommendations based on Google Trends data analysis.
    
    Parameters:
    - df (pd.DataFrame): Google Trends data with 'Date' and 'Interest Score'.
    - report (dict): Trends report containing trend summary, spikes, breakpoints, and forecast.
    - niche (str): The search term for which trends are analyzed.
    
    Returns:
    - list: List of actionable recommendations based on trend data.
    """
    recommendations = []

    # Analyze trend direction
    trend_summary = report.get('trend_summary', {})
    upward_trend_count = trend_summary.get('Upward', 0)
    downward_trend_count = trend_summary.get('Downward', 0)

    # Recent spikes and forecasted growth
    spikes = report.get('spike_events', {})
    forecast = report.get('forecast', {})
    breakpoints = report.get('breakpoints', [])

    # 1. General Trend Recommendations
    if upward_trend_count > downward_trend_count:
        recommendations.append(f"The overall trend for '{niche}' is upward. Consider increasing your marketing or expanding product offerings.")
    else:
        recommendations.append(f"The overall trend for '{niche}' is downward. Consider adjusting your strategy or exploring alternative niches.")

    # 2. Spike-Based Recommendations
    if len(spikes) > 0:
        spike_dates = list(spikes.keys())
        recommendations.append(f"Spikes detected on these dates: {', '.join(spike_dates)}. Consider launching short-term marketing campaigns during future similar trends.")

    # 3. Breakpoint-Based Recommendations
    if len(breakpoints) > 0:
        recommendations.append(f"Significant changes (breakpoints) detected in the trend. Consider reviewing the major shifts in interest to fine-tune your strategy.")

    # 4. Seasonal Recommendations
    if len(forecast) > 0:
        forecast_df = pd.DataFrame(forecast)
        # Filter for future data points
        future_forecast = forecast_df[forecast_df['ds'] >= datetime.now()]
        
        # Check the overall direction of future forecast
        forecast_trend = future_forecast['yhat'].diff().mean()
        if forecast_trend > 0:
            recommendations.append(f"Forecast shows an increasing interest in '{niche}' in the coming months. Prepare for an upward demand.")
        else:
            recommendations.append(f"Forecast suggests declining interest in '{niche}'. It may be wise to invest in alternative strategies or products.")

    # 5. Seasonal Influence
    current_month = datetime.now().month
    if current_month in [11, 12]:  # Consider the holiday season as a potential sales boost
        recommendations.append(f"The holiday season is approaching. Consider adjusting your offerings for increased demand during November and December.")

    # 6. Timeframe-Specific Actions
    if upward_trend_count > 0 and forecast_trend > 0:
        recommendations.append("Interest in this niche has been growing recently and is expected to continue rising. Accelerate your efforts now.")
    elif downward_trend_count > 0:
        recommendations.append(f"Recent trends for '{niche}' have shown a decline. It might be time to reevaluate your offerings or reposition them.")

    return recommendations

def interpret_google_trends(df, niche, forecast_period=90):
    """
    Analyze Google Trends data and provide trend summary, spikes, and forecast.
    
    Parameters:
    - df (pd.DataFrame): Google Trends data.
    - niche (str): Search term.
    - forecast_period (int): Days to forecast into the future.
    
    Returns:
    - dict: Dictionary containing trend summary, spikes, breakpoints, forecast, and recommendations.
    """
    df['Date'] = pd.to_datetime(df.index)
    df['Interest Score'] = df[niche].interpolate(method='linear')

    df['Trend'] = np.where(df['Interest Score'].diff() > 0, 'Upward', 'Downward')
    df['z_score'] = (df['Interest Score'] - df['Interest Score'].mean()) / df['Interest Score'].std()
    spikes = df[df['z_score'] > 2]

    algo = Pelt(model="rbf").fit(df['Interest Score'].values)
    breakpoints = algo.predict(pen=10)

    df_prophet = df[['Date', 'Interest Score']].rename(columns={'Date': 'ds', 'Interest Score': 'y'})
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    report = {
        'trend_summary': df['Trend'].value_counts().to_dict(),
        'spike_events': spikes[['Interest Score']].to_dict(),
        'breakpoints': breakpoints,
        'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict()
    }

    # Generate enhanced recommendations
    recommendations = generate_recommendations(df, report, niche)
    report['recommendations'] = recommendations

    return report

def get_trend_summary(report):
    """Return trend summary section."""
    trend_summary = report.get('trend_summary', {})
    return json.dumps(trend_summary, indent=4)

# TIMESTAMP CONVERTER / JSON CONVERTERS
def convert_timestamps_to_strings(data):
    """Converts Timestamp keys in a dictionary to string format 'YYYY-MM-DD'."""
    formatted_data = {}
    
    for key, value in data.items():
        if isinstance(key, pd.Timestamp):
            formatted_key = key.strftime('%Y-%m-%d')
        else:
            formatted_key = str(key)
        
        # Handle nested dictionaries or lists
        if isinstance(value, dict):
            value = convert_timestamps_to_strings(value)
        elif isinstance(value, list):
            value = [convert_timestamps_to_strings(item) if isinstance(item, dict) else item for item in value]
        
        formatted_data[formatted_key] = value
    
    return formatted_data

def get_spike_events(report):
    """Return spike events formatted for the frontend with Timestamps converted to strings."""
    spike_events = report.get('spike_events', {})

    formatted_spikes = convert_timestamps_to_strings(spike_events)
    return json.dumps(formatted_spikes, indent=4)

def get_breakpoints(report):
    """Return breakpoints formatted for the frontend."""
    breakpoints = report.get('breakpoints', [])
    formatted_breakpoints = [{'breakpoint_index': idx} for idx in breakpoints]
    return json.dumps(formatted_breakpoints, indent=4)

def get_recommendations(report):
    """Return recommendations."""
    recommendations = report.get('recommendations', [])
    return json.dumps(recommendations, indent=4)

def serve_requested_data(report, request_type):
    """
    Serve the requested section of the report based on the type requested by the frontend.

    Parameters:
    - report (dict): The full report dictionary.
    - request_type (str): The specific section requested by the frontend.

    Returns:
    - str: JSON formatted response for the requested data.
    """
    if request_type == 'trend_summary':
        return get_trend_summary(report)
    elif request_type == 'spike_events':
        return get_spike_events(report)
    elif request_type == 'breakpoints':
        return get_breakpoints(report)
    elif request_type == 'recommendations':
        return get_recommendations(report)
    else:
        return json.dumps({'error': 'Invalid request type'}, indent=4)

def compile_all_json(report):
    """Compile all JSON data into one object for the frontend."""
    all_data = {
        'trend_summary': json.loads(get_trend_summary(report)),
        'spike_events': json.loads(get_spike_events(report)),
        'breakpoints': json.loads(get_breakpoints(report)),
        'recommendations': json.loads(get_recommendations(report))
    }
    return json.dumps(all_data, indent=4)

def get_bar_graph_data(df, niche):
    """
    Prepare data for creating a bar graph showing interest scores over time.

    Parameters:
    - df (pd.DataFrame): Google Trends data.
    - niche (str): Search term.

    Returns:
    - dict: JSON-ready data for the bar graph with dates and interest scores.
    """
    df['Date'] = pd.to_datetime(df.index)
    bar_graph_data = df[['Date', niche]].dropna()

    # Convert the data into a list of dictionaries for easy frontend consumption
    bar_graph_json = {
        "labels": bar_graph_data['Date'].dt.strftime('%Y-%m-%d').tolist(),  # X-axis (dates)
        "values": bar_graph_data[niche].tolist()  # Y-axis (interest scores)
    }
    
    return json.dumps(bar_graph_json, indent=4)

def calculate_roi(df, niche, investment_amount=1000, forecast_period=90):
    """
    Calculate Return on Investment (ROI) based on Google Trends data and forecast.

    Parameters:
    - df (pd.DataFrame): Google Trends data.
    - niche (str): Search term.
    - investment_amount (float): The amount of money being invested (default is $1000).
    - forecast_period (int): The number of days into the future to forecast.

    Returns:
    - dict: Dictionary containing ROI estimation, growth trend, and suggestions.
    """
    df['Date'] = pd.to_datetime(df.index)
    df['Interest Score'] = df[niche].interpolate(method='linear')
    df_prophet = df[['Date', 'Interest Score']].rename(columns={'Date': 'ds', 'Interest Score': 'y'})

    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    future_forecast = forecast[forecast['ds'] >= datetime.now()]
    initial_interest = df['Interest Score'].iloc[-1]
    final_forecasted_interest = future_forecast['yhat'].iloc[-1]

    interest_growth = (final_forecasted_interest - initial_interest) / initial_interest
    estimated_roi = investment_amount * (1 + interest_growth)

    # Convert values to standard Python int and float types
    growth_trend = "upward" if interest_growth > 0 else "downward"
    interest_growth_percentage = round(float(interest_growth * 100), 2)

    return {
        'niche': niche,
        'investment_amount': f"${float(investment_amount)}",  # Ensure float type
        'forecast_period_days': f"{int(forecast_period)} days",    # Ensure int type
        'initial_interest_score': round(float(initial_interest), 2),  # Ensure float type
        'forecasted_interest_score': round(float(final_forecasted_interest), 2),  # Ensure float type
        'estimated_roi': f"{round(float(estimated_roi), 2)}%",  # Ensure float type
        'interest_growth': f"{interest_growth_percentage}%",  # Already float type
        'growth_trend': growth_trend,
        'recommendation': f"Based on the trend, the interest in '{niche}' is expected to {growth_trend}. "
                          f"Your estimated ROI for a ${investment_amount} investment over {forecast_period} days is {round(float(estimated_roi), 2)}%. "
                          f"NOTE: This calculation is based on the percentage growth of the interest score over a period of time."
    }


def calculate_growth_rate(current_value, previous_value):
    if previous_value == 0:
        return 0  # To avoid division by zero
    return ((current_value - previous_value) / previous_value) * 100

def calculate_profit_margin(total_revenue, total_expenses):
    if total_revenue == 0:
        return 0  # To avoid division by zero
    return ((total_revenue - total_expenses) / total_revenue) * 100

def calculate_average_revenue(current_revenue, months):
    if months == 0:
        return 0  # To avoid division by zero
    return current_revenue / months

def fetch_profit_margin(company_symbol):
    try:
        stock = yf.Ticker(company_symbol)
        financials = stock.financials
        if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index:
            profit_margin = (financials.loc['Gross Profit'] / financials.loc['Total Revenue']).mean()
            return profit_margin * 100  # Convert to percentage
    except Exception as e:
        print(f"Error fetching profit margin for {company_symbol}: {e}")
    return None

def fetch_industry_benchmarks(industry):
    industry_stocks = {
        'Retail': ['WMT', 'AMZN', 'TGT'],
        'Tech': ['AAPL', 'MSFT', 'GOOGL'],
        'Healthcare': ['JNJ', 'PFE', 'MRK'],
        'Finance': ['JPM', 'BAC', 'GS'],
        'Energy': ['XOM', 'CVX', 'BP'],
        'Consumer Goods': ['PG', 'KO', 'PEP'],
        'Utilities': ['NEE', 'DUK', 'SO'],
        'Industrial': ['GE', 'BA', 'HON'],
        # Add more industries as needed
    }

    try:
        if industry in industry_stocks:
            # Fetch profit margins for individual stocks
            profit_margins = []
            for symbol in industry_stocks[industry]:
                margin = fetch_profit_margin(symbol)  # Custom function to fetch profit margin
                if margin is not None:
                    profit_margins.append(margin)
            
            # Calculate average profit margin or use default value if no data
            avg_profit_margin = sum(profit_margins) / len(profit_margins) if profit_margins else 20

            # Map industries to specific ETFs for growth rate calculations
            industry_etfs = {
                'Retail': 'XRT',       # Retail ETF
                'Tech': 'XLK',         # Tech ETF
                'Healthcare': 'XLV',   # Healthcare ETF
                'Finance': 'XLF',      # Finance ETF
                'Energy': 'XLE',       # Energy ETF
                'Consumer Goods': 'XLP',  # Consumer Staples ETF
                'Utilities': 'XLU',    # Utilities ETF
                'Industrial': 'XLI'    # Industrial ETF
            }
            
            # Determine the appropriate ETF ticker
            ticker = industry_etfs.get(industry, f"{industry} ETF")  # Fallback to placeholder if no specific ETF

            # Fetch historical data from Yahoo Finance
            data = yf.download(ticker, period="1y")
            
            # Calculate the average growth rate based on the closing prices
            avg_growth_rate = data['Close'].pct_change().mean() * 100  # Annualized growth rate

            return avg_growth_rate, avg_profit_margin
    except Exception as e:
        print(f"Error fetching benchmarks for {industry}: {e}")
    
    # Return default values in case of an error
    return 5, 15


def generate_business_insights(current_revenue, previous_revenue, total_expenses, customer_base, months, low_growth_rate_threshold, low_profit_margin_threshold):
    # Input validation
    if current_revenue < 0 or previous_revenue < 0 or total_expenses < 0 or customer_base < 0 or months <= 0:
        raise ValueError("Revenue, expenses, and customer base must be non-negative and months must be positive.")

    # Calculate metrics
    calculated_growth_rate = calculate_growth_rate(current_revenue, previous_revenue)
    profit_margin = calculate_profit_margin(current_revenue, total_expenses)
    average_revenue_per_month = calculate_average_revenue(current_revenue, months)

    # Define thresholds based on fetched benchmarks
    high_revenue_threshold = current_revenue * 2  # Example: double the current revenue
    medium_profit_margin_threshold = low_profit_margin_threshold + 10  # Adding a buffer for medium threshold

    suggestions = []

    # Revenue Analysis
    if current_revenue < 50000:
        suggestions.append("Your revenue is relatively low. Consider implementing aggressive marketing strategies to attract new customers.")
    elif current_revenue > high_revenue_threshold:
        suggestions.append("With a high revenue, ensure you are investing in sustainable growth and exploring expansion opportunities.")

    # Profit Margin Analysis
    if profit_margin < low_profit_margin_threshold:
        suggestions.append("Your profit margin is below industry standards. Focus on optimizing your supply chain and reducing operational costs.")
    elif profit_margin < medium_profit_margin_threshold:
        suggestions.append("A moderate profit margin suggests potential. Evaluate your pricing strategy and consider adding value-added services.")
    else:
        suggestions.append("Excellent profit margin! Consider reinvesting in innovation or enhancing customer experience to maintain this advantage.")

    # Growth Rate Analysis
    if calculated_growth_rate < 5:
        suggestions.append("A low growth rate indicates potential stagnation. Explore new market segments or diversify your product offerings.")
    elif calculated_growth_rate < low_growth_rate_threshold:
        suggestions.append("Your growth rate is healthy. Keep an eye on market trends and be ready to adapt to changes.")
    else:
        suggestions.append("Impressive growth! Consider scalability strategies to sustain this momentum.")

    # Average Revenue Analysis
    suggestions.append(f"The average revenue per month is ${average_revenue_per_month:.2f}.")

    # Customer Base Analysis
    if customer_base < 100:
        suggestions.append("A small customer base may limit your market impact. Focus on customer retention and word-of-mouth referrals.")
    elif customer_base < 500:
        suggestions.append("A growing customer base is encouraging. Consider loyalty programs to enhance customer retention.")
    else:
        suggestions.append("With a substantial customer base, explore opportunities for upselling and cross-selling products.")

    # General Recommendations
    suggestions.extend([
        "Regularly review your financial metrics to identify areas for improvement.",
        "Engage with your customers to gather feedback and adapt your offerings based on their needs.",
        "Consider leveraging data analytics to better understand market trends and customer behavior."
    ])

    return {
        "suggestions": suggestions,
        "growth_rate": calculated_growth_rate,
        "profit_margin": profit_margin,
        "average_revenue_per_month": average_revenue_per_month
    }


def get_interest_by_region(niche, timeframe='today 12-m', location='US'):
    """
    Fetch and aggregate Google Trends data by region for a given niche.
    
    Parameters:
    - niche (str): Search term for Google Trends.
    - timeframe (str): Time range for trends (default is 1 year).
    - location (str): Location for the trends data.

    Returns:
    - dict: JSON-ready data for pie chart with regions and their corresponding interest scores.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([niche], timeframe=timeframe, geo=location)
    
    # Fetch regional interest data
    regional_data = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)

    # Filter and prepare the data
    if regional_data.empty:
        return {'error': 'No data found for the specified niche and location.'}

    # Reset index to have regions as a column
    regional_data.reset_index(inplace=True)
    
    # Convert to JSON-friendly format for pie chart
    pie_chart_data = {
        'labels': regional_data['geoName'].tolist(),  # Regions
        'values': regional_data[niche].tolist()  # Interest scores
    }

    return json.dumps(pie_chart_data, indent=4)


def get_current_month_interest_function(niche='iced coffee', location='US', timeframe='today 12-m'):
    """
    Retrieves the interest score for the current month for a given niche.
    
    Args:
        niche (str): Search term for Google Trends.
        location (str): Location code for the trends (e.g., US, PH).
        timeframe (str): Time range for trends (default is 1 year).
        
    Returns:
        dict: Interest score for the current month or an error message.
    """
    # Fetch Google Trends data
    trending_data = get_google_trends_data(niche, timeframe=timeframe, location=location)

    if trending_data is None:
        return {'error': 'No data found for the given niche or location.'}, 404

    # Calculate interest for the current month
    current_month_interest = get_current_month_interest(trending_data, niche)

    return current_month_interest

def trend_seeker(keyword, location, timeframe, top_n=5):
    """
    Retrieves the top keyword suggestions based on the provided keyword.

    Args:
        keyword (str): A keyword to analyze.
        location (str): Location code for trends (e.g., US, PH).
        timeframe (str): Time range for trends.
        top_n (int): The number of top suggestions to return.

    Returns:
        dict: A dictionary containing the suggestions or an error message.
    """
    pytrends = TrendReq(hl='en-US', tz=360)

    try:
        # Get keyword suggestions
        suggestions_data = pytrends.suggestions(keyword=keyword)
        if not suggestions_data:
            return {"error": "No suggestions found for the given keyword."}

        # Extract the top N titles from suggestions
        suggested_keywords = [suggestion['title'] for suggestion in suggestions_data[:top_n]]
        
        error_message = "No Data Found"
        
        results = []
        for niche in suggested_keywords:
            tries = 3
            while True:  # Loop for retrying on failure
                try:
                    # Fetch current month interest data for the niche
                    niche_data = get_current_month_interest_function(niche, location, timeframe)
                    results.append({
                        'keyword': niche,
                        'niche_data': niche_data,  # Directly append the niche_data as it is already a dict
                    })
                    break  # Break the loop if successful

                except Exception as e:
                    if '429' in str(e):
                        tries -= 1
                        if tries == 0:
                            return {
                                'error': error_message,
                                'suggested_keywords': suggested_keywords
                            }
                        time.sleep(3)  # Longer wait time for rate limiting
                    else:
                        return {"error": str(e)}  # For other exceptions

            # Sleep for 5 seconds between requests to avoid hitting rate limits
            time.sleep(5)  # Adjust as needed
            
        return {
            'original_keyword': keyword,
            'top_suggestions': results
        }

    except Exception as e:
        return {"error": str(e)}
