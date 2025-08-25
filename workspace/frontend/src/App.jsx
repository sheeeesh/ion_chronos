import React from "react";
import JobForm from "./components/JobForm";
import JobList from "./components/JobList";
import "./styles.css";
export default function App(){
  return (
    <div className="app">
      <header className="topbar"><h1>Ion Chronos</h1></header>
      <div className="container">
        <div className="left"><JobForm/></div>
        <div className="right"><JobList/></div>
      </div>
    </div>
  );
}
