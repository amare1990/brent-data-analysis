import React, { useEffect, useState } from "react";
import { fetchArimaResults } from "../services/api";


const ArimaResults = () => {
    const [arimaSummary, setArimaSummary] = useState("");
        const [error, setError] = useState(null);

        useEffect(() => {
            async function getData() {
                try {
                    const arima = await fetchArimaResults();
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

export default ArimaResults;
