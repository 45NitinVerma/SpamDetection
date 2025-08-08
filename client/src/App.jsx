import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [message, setMessage] = useState("");
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);

  const appUrl = import.meta.env.VITE_APP_URL || "http://localhost:5000";

  const handlePredict = async () => {
    if (!message.trim()) return alert("Please enter a message");

    setLoading(true);
    try {
      const response = await axios.post(`${appUrl}/predict`, {
        message,
      });

      if (response.data?.prediction) {
        setPrediction(response.data.prediction);
      } else {
        setPrediction("❌ Error: Invalid response");
      }
    } catch (error) {
      console.error(error);
      setPrediction("❌ Error: Could not connect to server");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Spam Detection App</h1>
      <textarea
        placeholder="Enter email or SMS text..."
        rows="6"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      ></textarea>
      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>
      {prediction && <div className="result">Result: {prediction}</div>}
    </div>
  );
}

export default App;
