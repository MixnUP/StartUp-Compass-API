import pandas as pd
import logging
from services import (
    get_interest_by_region, 
    interpret_google_trends, 
    get_bar_graph_data,
    trend_seeker,
    get_google_trends_data  # Add this import
)
from typing import Dict, Any

def comprehensive_trend_analysis(
    api_data: Dict[str, str]
) -> Dict[str, Any]:
    """
    Perform comprehensive trend analysis based on API parameters.
    
    Args:
        api_data (dict): API parameters for trend analysis
    
    Returns:
        dict: Comprehensive trend analysis report
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Extract key parameters
    industry = api_data.get('industry', 'technology')
    location = api_data.get('location', 'US')
    business_scale = api_data.get('business_scale', 'medium')

    # Trend analysis results
    trend_analysis = {
        'error': None,
        'regional_trends': {},
        'trend_forecast': {},
        'trend_data': {
            'yearly': {},
            'monthly': {},
            'weekly': {},
            'daily': {}
        },
        'related_keywords': [],
        'trend_chart': {}
    }

    try:
        # Attempt to get Google Trends data
        try:
            trends_df = get_google_trends_data(
                niche=industry, 
                location=location, 
                timeframe='today 12-m'
            )
            
            if trends_df is None or trends_df.empty:
                raise ValueError("No trend data retrieved")

        except Exception as trend_error:
            logger.error(f"Failed to retrieve Google Trends data: {trend_error}")
            trend_analysis['error'] = f"Google Trends data retrieval failed: {str(trend_error)}"
            trends_df = pd.DataFrame()  # Ensure we have an empty DataFrame

        # Get regional interest data
        try:
            trend_analysis['regional_trends'] = get_interest_by_region(
                niche=industry, 
                location=location
            )
        except Exception as region_error:
            logger.warning(f"Failed to retrieve regional trends: {region_error}")
            trend_analysis['regional_trends'] = {}

        # Trend forecast and data
        try:
            trend_report = interpret_google_trends(trends_df, industry)
            
            # Populate trend data
            trend_analysis['trend_forecast'] = trend_report.get('forecast', {})
            trend_analysis['trend_data'] = {
                'yearly': trend_report.get('yearly_trends', {}),
                'monthly': trend_report.get('monthly_trends', {}),
                'weekly': trend_report.get('weekly_trends', {}),
                'daily': trend_report.get('daily_trends', {})
            }
        except Exception as interpret_error:
            logger.warning(f"Failed to interpret trends: {interpret_error}")
            trend_analysis['error'] = f"Trend interpretation failed: {str(interpret_error)}"

        # Bar graph data for trends
        try:
            trend_analysis['trend_chart'] = get_bar_graph_data(trends_df, industry)
        except Exception as chart_error:
            logger.warning(f"Failed to generate trend chart: {chart_error}")
            trend_analysis['trend_chart'] = {}

        # Find related keywords
        try:
            related_keywords_result = trend_seeker(
                keyword=industry, 
                location=location, 
                timeframe='today 12-m', 
                top_n=5
            )
            trend_analysis['related_keywords'] = related_keywords_result.get('suggestions', [])
        except Exception as keyword_error:
            logger.warning(f"Failed to retrieve related keywords: {keyword_error}")
            trend_analysis['related_keywords'] = []

    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        trend_analysis['error'] = str(e)

    # Add metadata
    trend_analysis['metadata'] = {
        'industry': industry,
        'location': location,
        'business_scale': business_scale,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }

    return trend_analysis