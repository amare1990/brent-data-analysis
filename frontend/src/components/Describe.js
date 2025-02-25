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
                    Object.keys(dataDescription).map((column, index) => (
                        <div key={index} className="overview-content">
                            <h4>{column}</h4>
                            <p><strong>Count:</strong> {dataDescription[column].count}</p>
                            <p><strong>Mean:</strong> {dataDescription[column].mean}</p>
                            <p><strong>Standard Deviation:</strong> {dataDescription[column].std}</p>
                            <p><strong>Min:</strong> {dataDescription[column].min}</p>
                            <p><strong>25th Percentile (Q1):</strong> {dataDescription[column]["25%"]}</p>
                            <p><strong>Median (50th Percentile):</strong> {dataDescription[column]["50%"]}</p>
                            <p><strong>75th Percentile (Q3):</strong> {dataDescription[column]["75%"]}</p>
                            <p><strong>Max:</strong> {dataDescription[column].max}</p>
                        </div>
                    ))
                ) : (
                    <p>Loading description...</p>
                )}
            </div>
        </div>
    );
};

export default Describe;
