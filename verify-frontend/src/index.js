import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
import { createBrowserRouter, Link, RouterProvider } from "react-router-dom";
import CreateUser from "./CreateUser";

const router = createBrowserRouter([
    {
        path: "/",
        element: <App />,
        errorElement: (
            <div className="flex flex-col gap-2 text-xl m-10">
                404 Not Found.
                <Link to="/" className="text-blue-700 font-bold">
                    Go back home
                </Link>
            </div>
        ),
    },
    { path: "/createUser", element: <CreateUser /> },
]);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
    <React.StrictMode>
        <RouterProvider router={router} />
    </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
