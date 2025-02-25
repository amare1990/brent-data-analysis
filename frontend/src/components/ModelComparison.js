import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { fetchModelComparison } from "../services/api";
import './ModelComparison.css';  // CSS for styling

const ModelComparison = () => {
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const data = await fetchModelComparison();
      console.log("📊 Setting models state:", data);
      setMetrics(data);
    };

    fetchData();
  }, [])

  return (
    <div className="container">
      <h1>Model Performance Comparison</h1>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>R² Score</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((model, index) => (
            <tr key={index}>
              <td>{model.Model}</td>
              <td>{model.RMSE ? model.RMSE.toFixed(4) : "N/A"}</td>
              <td>{model.MAE ? model.MAE.toFixed(4) : "N/A"}</td>
              <td>{model["R² Score"] ? model["R² Score"].toFixed(4) : "N/A"}</td>

            </tr>
          ))}
        </tbody>
      </table>
      <div className="notes">
        <p>✅ Model with lowest RMSE and MAE is generally the best.</p>
        <p>✅ Higher R² Score (closer to 1) indicates better model fit.</p>
      </div>
    </div>
  );
};

export default ModelComparison;
