import React, { useEffect, useState } from "react";
import { fetchDescribeData } from "../services/api";
import "./Describe.css"; // Import CSS for styling

const Describe = () => {
    const [dataDescription, setDataDescription] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function getData() {
            try {
                const description = await fetchDescribeData();
                if (description) setDataDescription(description);
            } catch (err) {
                console.error("API Error:", err);
                setError("Error fetching data description.");
            }
        }
        getData();
    }, []);

    return (
        <div className="overview-container">
            <h2>📊 Data Statistical Overview</h2>
            {error ? <p className="error">{error}</p> : null}

            <div className="overview-section">
                <h3>🔍 Statistical Description of Data</h3>
                {dataDescription ? (
                    <div className="overview-content">
                        <p><strong>Count:</strong> {dataDescription.count}</p>
                        <p><strong>Mean:</strong> {dataDescription.mean}</p>
                        <p><strong>Standard Deviation:</strong> {dataDescription.std}</p>
                        <p><strong>Min:</strong> {dataDescription.min}</p>
                        <p><strong>25th Percentile (Q1):</strong> {dataDescription["25%"]}</p>
                        <p><strong>Median (50th Percentile):</strong> {dataDescription["50%"]}</p>
                        <p><strong>75th Percentile (Q3):</strong> {dataDescription["75%"]}</p>
                        <p><strong>Max:</strong> {dataDescription.max}</p>
                    </div>
                ) : (
                    <p>Loading description...</p>
                )}
            </div>
        </div>
    );
};

export default Describe;
