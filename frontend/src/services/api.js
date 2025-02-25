const BASE_URL = "http://localhost:5000";

export async function fetchOilData() {
    try {
        const response = await fetch(`${BASE_URL}/api/data`);
        return await response.json();
    } catch (error) {
        console.error("Error fetching oil data:", error);
        return null;
    }
}

export async function fetchArimaSummary() {
    try {
        const response = await fetch(`${BASE_URL}/api/arima`);
        return await response.json();
    } catch (error) {
        console.error("Error fetching ARIMA summary:", error);
        return null;
    }
}
