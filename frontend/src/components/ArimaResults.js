import React, { useEffect, useState } from "react";
import { fetchArimaResults } from "../services/api";

const ArimaResults = () => {
    const [arimaResults, setArimaResults] = useState(null);

    useEffect(() => {
        async function getResults() {
            const results = await fetchArimaResults();
            setArimaResults(results);
        }
        getResults();
    }, []);

    return (
        <div>
            <h2>ARIMA Summary</h2>
            {arimaResults ? (
                <pre>{JSON.stringify(arimaResults, null, 2)}</pre>
            ) : (
                <p>Loading...</p>
            )}
        </div>
    );
};

export default ArimaResults;
