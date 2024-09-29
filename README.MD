## StartUp Compass API Documentation

This README provides a brief overview of the StartUp Compass API documentation, which you can find [here](https://startup-compass-api.onrender.com).

**The documentation covers the following API endpoints:**

* **GET /get_google_trends:** Fetch Google Trends data and return specific sections like trend summary, spike events, breakpoints, and recommendations.
* **GET /get_current_month_interest:** Retrieve the current month's interest score for a specific niche based on Google Trends data.
* **GET /compile_all:** Compile all sections of a Google Trends report into a single JSON response.
* **GET /calculate_roi:** Estimate the Return on Investment (ROI) based on Google Trends forecast for a specific niche.
* **GET /get-bar-graph-data:** Retrieve data for creating a bar graph showing interest scores over time for a specific niche.
* **GET /business-assessment:** Submit business metrics to receive tailored insights and recommendations based on financial performance.
* **GET /interest_by_region:** Retrieve Google Trends data for a specified niche, broken down by region, to visualize interest levels using a pie chart.
* **GET /trend_seeker:** Retrieve top keyword suggestions based on a specified keyword along with their current month interest data.

**Key Features of the API Documentation:**

* **Detailed Endpoint Descriptions:** Each section provides a clear explanation of the endpoint's purpose, parameters, and expected responses.
* **Example Requests and Responses:**  Illustrative examples demonstrate how to use the API with specific query parameters and show the expected JSON output.
* **Error Handling:**  Information on potential error responses and their causes is included.

**Important Note:** The trends API utilizes the `pytrends` library version 4.9.2.

**To use the API:**
1. Refer to the documentation for each endpoint to understand its functionality and parameters.
2. Construct your API requests using the provided examples as a guide.
3. Interpret the JSON responses to gain valuable insights and data for your business. 

