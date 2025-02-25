import axios from "axios";

// Set Flask backend URL
const BASE_URL = "http://localhost:5000";

// Function to fetch Brent oil data
export const fetchOilData = async () => {
    try {
        const response = await axios.get(`${BASE_URL}/api/data`);
        return response.data;
    } catch (error) {
        console.error("Error fetching Brent oil data:", error);
        return null;
    }
};


// Function to fetch data description (statistical summary)
export const fetchDescribeData = async () => {
  try {
      const response = await axios.get(`${BASE_URL}/api/describe`);
      return response.data; // Returns the statistical description (JSON)
  } catch (error) {
      console.error("Error fetching data description:", error);
      return null;
  }
};

// Function to fetch ARIMA results
export const fetchArimaResults = async () => {
    try {
        const response = await axios.get(`${BASE_URL}/api/arima`);
        return response.data;
    } catch (error) {
        console.error("Error fetching ARIMA results:", error);
        return null;
    }
};
