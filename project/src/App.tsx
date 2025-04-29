import { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [data, setData] = useState<string>('');

  useEffect(() => {
    // Example: fetch from Streamlit backend
    axios.get('http://localhost:8501')
      .then(response => {
        setData('Backend is running!');
      })
      .catch(error => {
        setData('Backend not reachable.');
      });
  }, []);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">My Project Dashboard</h1>
        <p className="text-xl">{data}</p>
      </div>
    </div>
  );
}

export default App;
