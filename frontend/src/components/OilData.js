import React, { useEffect, useState } from "react";
import { fetchOilData, fetchArimaSummary } from "../services/api";
import "./OilData.css"; // Import CSS

const OilData = () => {
    const [oilData, setOilData] = useState(null);
    const [arimaSummary, setArimaSummary] = useState("");
    const [error, setError] = useState(null);

    useEffect(() => {
        async function getData() {
            try {
                const data = await fetchOilData();
                const arima = await fetchArimaSummary();
                if (data) setOilData(data);
                if (arima) setArimaSummary(arima.arima_summary);
            } catch (err) {
                console.error("API Error:", err);
                setError("Error fetching data.");
            }
        }
        getData();
    }, []);

    return (
        <div className="container">
            <h2>📈 Brent Oil Data</h2>
            {error ? <p className="error">{error}</p> : null}

            <div className="section">
                <h3>🔍 Oil Data</h3>
                {oilData ? (
                    <pre className="json-box">{JSON.stringify(oilData, null, 2)}</pre>
                ) : (
                    <p>Loading data...</p>
                )}
            </div>

            <div className="section">
                <h3>📊 ARIMA Summary</h3>
                {arimaSummary ? (
                    <pre className="summary-box">{arimaSummary}</pre>
                ) : (
                    <p>Loading ARIMA summary...</p>
                )}
            </div>
        </div>
    );
};

export default OilData;
