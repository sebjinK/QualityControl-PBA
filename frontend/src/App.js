import React from 'react';
import { Navbar, Container, Nav, Button } from 'react-bootstrap';

// This App component is now minimal, clean, and ready for you to build your content quickly.
function App() {
  return (
    <div className="App bg-light h-screen">
      {/* This Navbar is a standard React-Bootstrap component.
        It should inherit the existing Hope UI styles if the CSS 
        is still correctly layered over Bootstrap's defaults.
      */}
      <Navbar bg="dark" variant="dark" expand="lg" className="shadow-md">
        <Container>
          <Navbar.Brand href="#home">
            <span className="font-bold text-xl text-yellow-400">
              Portobello America
            </span>
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link href="#qualityControl">Quality Control</Nav.Link>
            </Nav>
            <Nav>
              <Button variant="outline-light" className="hover:bg-yellow-500 hover:text-black transition-all duration-200">
                Example Button
              </Button>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      {/* Main content area below the Navbar */}
      <Container className="my-5 p-4 rounded-lg bg-white shadow-lg">
        <h1 className="text-4xl font-light mb-4">Quality Control</h1>
        <p className="text-lg text-gray-700">
          Start building here
        </p>
      </Container>
    </div>
  );
}

export default App;
