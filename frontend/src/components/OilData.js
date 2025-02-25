import React, { useEffect, useState } from "react";
import { fetchOilData } from "../services/api";

const OilData = () => {
    const [oilData, setOilData] = useState(null);

    useEffect(() => {
        async function getData() {
            const data = await fetchOilData();
            setOilData(data);
        }
        getData();
    }, []);

    return (
        <div>
            <h2>Brent Oil Data</h2>
            {oilData ? (
                <pre>{JSON.stringify(oilData, null, 2)}</pre>
            ) : (
                <p>Loading...</p>
            )}
        </div>
    );
};

export default OilData;
