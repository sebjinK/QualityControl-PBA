import React from "react";

{ /*
    Filename:    NoPage.js
    Description: This is Bridging Hope's 404 page. It is displayed when a user navigates to a page that does not exist.
*/ }

const NoPage = () => {
    return (
        <React.Fragment>
            <h1 className="text-center mt-3">404 Page not found</h1>
            <a href="/" className="d-flex justify-content-center">Take me to the home page</a>
        </React.Fragment>
    );
};

export default NoPage;
