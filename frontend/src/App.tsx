import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [backendStatus, setBackendStatus] = useState<string>("Loading...");
  const [backendData, setBackendData] = useState<any>(null);

  useEffect(() => {
    console.log("Fetching backend health...");

    fetch("http://127.0.0.1:8000/health")
      .then((res) => {
        console.log("Response:", res);
        return res.json();
      })
      .then((data) => {
        console.log("Parsed data:", data);

        setBackendData(data);
        setBackendStatus(`${data.status} - ${data.message}`);
      })
      .catch((err) => {
        console.error("Error:", err);
        setBackendStatus("Error: cannot reach backend");
      });
  }, []);

  useEffect(() => {
    console.log("backendStatus updated:", backendStatus);
  }, [backendStatus]);

  return (
    <div style={{ fontFamily: "sans-serif", padding: "2rem" }}>
      <h1>My Quant Platform</h1>
      <p>Backend status: {backendStatus}</p>

      <h3>Debug output:</h3>
      <pre>{JSON.stringify(backendData, null, 2)}</pre>
    </div>
  );
}

export default App;

