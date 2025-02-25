import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './ModelComparison.css';  // CSS for styling

const ModelComparison = () => {
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    // Fetch model comparison data from Flask backend
    axios.get('/api/compare_models')
      .then(response => {
        setMetrics(response.data);
      })
      .catch(error => {
        console.error("There was an error fetching the data!", error);
      });
  }, []);

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
              <td>{model.model}</td>
              <td>{model.rmse.toFixed(4)}</td>
              <td>{model.mae.toFixed(4)}</td>
              <td>{model.r2.toFixed(4)}</td>
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
