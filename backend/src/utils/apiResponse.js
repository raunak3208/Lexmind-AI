const success = (res, data = {}, message = "OK", statusCode = 200) => {
  return res.status(statusCode).json({
    success: true,
    message,
    data,
  });
};

const error = (res, message = "Something went wrong", statusCode = 500, details = null) => {
  const body = { success: false, message };
  if (details && process.env.NODE_ENV !== "production") {
    body.details = details;
  }
  return res.status(statusCode).json(body);
};

const created = (res, data = {}, message = "Created") => success(res, data, message, 201);

module.exports = { success, error, created };