// Require necessary packages
const express = require("express");
const path = require("path");
require("dotenv").config();
const mysql = require("mysql2");
const cors = require("cors"); // CORS middleware
const jwt = require("jsonwebtoken"); // JWT for token generation

// Initialize the Express app
const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true })); // Ensure URL-encoded data is parsed
app.use(cors()); // Enable CORS to handle requests from different origins
app.use(express.static(path.join(__dirname, "templates")));

// Database Connection
const db = mysql.createConnection({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
});

db.connect((err) => {
  if (err) {
    console.error("Database connection failed:", err);
    process.exit(1);
  }
  console.log("Connected to the MySQL database");
});

// Route for the homepage (serve index.html by default)
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "templates", "index.html"));
});

// User Registration Route (Sign-up)
app.post("/registration", async (req, res) => {
  console.log("Received request body:", req.body); // Debugging line

  const { name, email, reg_no, password } = req.body;

  // Validate the fields
  if (!name || !reg_no || !email || !password) {
    console.log("Missing fields!"); // Log missing fields
    return res
      .status(400)
      .json({ message: "Please provide all required fields." });
  }

  try {
    // Store user in the database without hashing
    const query = `INSERT INTO doc_auth (name, email, reg_no, password) VALUES (?, ?, ?, ?)`;

    db.query(query, [name, email, reg_no, password], (err, result) => {
      if (err) {
        console.error("Database Error:", err);
        return res.status(500).json({ message: "Database error", error: err });
      }
      // res.status(201).json({ message: "User registered successfully" });
      res.redirect("/index.html");
    });
  } catch (err) {
    console.error("Registration Error:", err);
    res
      .status(500)
      .json({ message: "Error while registering user", error: err });
  }
});

// User Login Route (Sign-in)
app.post("/login", async (req, res) => {
  console.log("Received login request:", req.body);

  const { email, password } = req.body;

  // Validate input fields
  if (!email || !password) {
    console.log("Missing email or password!");
    return res.status(400).json({ message: "Please provide both email and password." });
  }

  try {
    // Step 1: Check if the email exists
    const query = `SELECT * FROM doc_auth WHERE email = ?`;

    db.query(query, [email], (err, results) => {
      if (err) {
        console.error("Database Error:", err);
        return res.status(500).json({ message: "Database error", error: err });
      }

      if (results.length === 0) {
        console.log("Invalid login attempt: Email not found");
        return res.status(401).json({ message: "Invalid email or password." });
      }

      // Step 2: User exists, now check password
      const user = results[0]; // Get user details from DB
      const storedPassword = user.password; // Fetch stored password
      const userName = user.name; // Fetch user name

      if (storedPassword !== password) {
        console.log("Invalid login attempt: Incorrect password");
        return res.status(401).json({ message: "Invalid email or password." });
      }

      // Step 3: Successful login
      console.log("User Logged In:");
      console.log("Name:", userName);
      console.log("Email:", email);
      console.log("Login successful");

      // Step 4: Send response with user name for frontend storage
      res.redirect('/dashboard.html'); // Redirects immediately
    });
  } catch (err) {
    console.error("Login Error:", err);
    res.status(500).json({ message: "Error while logging in", error: err });
  }
});



// Start server
const PORT = process.env.PORT || 5010;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});