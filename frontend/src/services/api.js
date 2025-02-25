import axios from "axios";

// Set Flask backend URL
const BASE_URL = "http://127.0.0.1:5000";

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
