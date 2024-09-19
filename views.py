from flask import Blueprint, request, jsonify, render_template
from services import *

views = Blueprint(__name__, "views")


@views.route('/')
def home():
    return render_template("index.html")

@views.route('/get_google_trends', methods=['GET'])
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

@views.route('/get_current_month_interest', methods=['GET'])
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

@views.route('/compile_all', methods=['GET'])
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

@views.route('/get-bar-graph-data', methods=['GET'])
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

@views.route('/calculate_roi', methods=['GET'])
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


