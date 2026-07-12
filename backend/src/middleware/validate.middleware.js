const { validationResult } = require("express-validator");
const { error } = require("../utils/apiResponse");

// Collects express-validator results and responds with the first error message,
// replacing the duplicated validationResult checks across routes.
const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return error(res, errors.array()[0].msg, 422);
  }
  return next();
};

module.exports = validate;
