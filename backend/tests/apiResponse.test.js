const { created, error, success } = require("../src/utils/apiResponse");

const makeResponse = () => {
  const res = {
    status: jest.fn(),
    json: jest.fn(),
  };
  res.status.mockReturnValue(res);
  res.json.mockReturnValue(res);
  return res;
};

describe("apiResponse", () => {
  const originalNodeEnv = process.env.NODE_ENV;

  afterEach(() => {
    if (originalNodeEnv === undefined) {
      delete process.env.NODE_ENV;
    } else {
      process.env.NODE_ENV = originalNodeEnv;
    }
  });

  test("success sends the default response envelope", () => {
    const res = makeResponse();

    const result = success(res);

    expect(res.status).toHaveBeenCalledWith(200);
    expect(res.json).toHaveBeenCalledWith({
      success: true,
      message: "OK",
      data: {},
    });
    expect(result).toBe(res);
  });

  test("created delegates to a 201 success response", () => {
    const res = makeResponse();
    const data = { id: "document-1" };

    created(res, data, "Document created");

    expect(res.status).toHaveBeenCalledWith(201);
    expect(res.json).toHaveBeenCalledWith({
      success: true,
      message: "Document created",
      data,
    });
  });

  test("error includes details outside production", () => {
    process.env.NODE_ENV = "test";
    const res = makeResponse();

    error(res, "Validation failed", 422, { field: "email" });

    expect(res.status).toHaveBeenCalledWith(422);
    expect(res.json).toHaveBeenCalledWith({
      success: false,
      message: "Validation failed",
      details: { field: "email" },
    });
  });

  test("error omits details in production", () => {
    process.env.NODE_ENV = "production";
    const res = makeResponse();

    error(res, "Unexpected failure", 500, { secret: "internal" });

    expect(res.json).toHaveBeenCalledWith({
      success: false,
      message: "Unexpected failure",
    });
  });
});
