const mongoose = require("mongoose");

const connectDB = async () => {
  const uri = process.env.MONGO_URI;

  if (!uri) {
    console.error(" MONGO_URI is not set in .env");
    process.exit(1);
  }

  try {
    await mongoose.connect(uri, {
      serverSelectionTimeoutMS: 5000,
    });
    console.log(`MongoDB connected: ${mongoose.connection.host}`);
  } catch (err) {
    console.error(`MongoDB connection failed: ${err.message}`);
    // Retry once after 3 seconds before exiting
    setTimeout(async () => {
      try {
        await mongoose.connect(uri);
        console.log("MongoDB reconnected");
      } catch (e) {
        console.error("Retry failed. Exiting.");
        process.exit(1);
      }
    }, 3000);
  }
};

module.exports = connectDB;