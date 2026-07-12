const {
  ownerOrLawyer,
  requireLevel,
  restrictTo,
} = require("../src/middleware/rbac.middleware");

const makeResponse = () => {
  const res = {
    status: jest.fn(),
    json: jest.fn(),
  };
  res.status.mockReturnValue(res);
  res.json.mockReturnValue(res);
  return res;
};

const expectError = (res, statusCode, message) => {
  expect(res.status).toHaveBeenCalledWith(statusCode);
  expect(res.json).toHaveBeenCalledWith({
    success: false,
    message,
  });
};

describe("restrictTo", () => {
  test("rejects unauthenticated requests", () => {
    const res = makeResponse();
    const next = jest.fn();

    restrictTo("lawyer")({}, res, next);

    expectError(res, 401, "Not authenticated");
    expect(next).not.toHaveBeenCalled();
  });

  test("rejects users outside the allowed roles", () => {
    const res = makeResponse();
    const next = jest.fn();

    restrictTo("lawyer", "paralegal")(
      { user: { role: "client" } },
      res,
      next
    );

    expectError(res, 403, "Access denied. Required role: lawyer or paralegal");
    expect(next).not.toHaveBeenCalled();
  });

  test("passes allowed users to the next middleware", () => {
    const next = jest.fn();

    restrictTo("lawyer")(
      { user: { role: "lawyer" } },
      makeResponse(),
      next
    );

    expect(next).toHaveBeenCalledTimes(1);
  });
});

describe("requireLevel", () => {
  test("rejects unauthenticated requests", () => {
    const res = makeResponse();

    requireLevel("client")({}, res, jest.fn());

    expectError(res, 401, "Not authenticated");
  });

  test("rejects an invalid minimum role", () => {
    const res = makeResponse();

    requireLevel("administrator")(
      { user: { role: "lawyer" } },
      res,
      jest.fn()
    );

    expectError(res, 500, "Invalid role configuration");
  });

  test("rejects users below the minimum role", () => {
    const res = makeResponse();

    requireLevel("paralegal")(
      { user: { role: "client" } },
      res,
      jest.fn()
    );

    expectError(res, 403, "Access denied. Minimum role required: paralegal");
  });

  test("allows users at or above the minimum role", () => {
    const next = jest.fn();

    requireLevel("paralegal")(
      { user: { role: "lawyer" } },
      makeResponse(),
      next
    );

    expect(next).toHaveBeenCalledTimes(1);
  });
});

describe("ownerOrLawyer", () => {
  test("rejects unauthenticated requests", async () => {
    const res = makeResponse();

    await ownerOrLawyer(jest.fn())({}, res, jest.fn());

    expectError(res, 401, "Not authenticated");
  });

  test("allows lawyers without loading resource ownership", async () => {
    const getResourceUserId = jest.fn();
    const next = jest.fn();

    await ownerOrLawyer(getResourceUserId)(
      { user: { role: "lawyer" } },
      makeResponse(),
      next
    );

    expect(getResourceUserId).not.toHaveBeenCalled();
    expect(next).toHaveBeenCalledTimes(1);
  });

  test("returns not found when the resource has no owner", async () => {
    const res = makeResponse();

    await ownerOrLawyer(async () => null)(
      { user: { role: "client", _id: "user-1" } },
      res,
      jest.fn()
    );

    expectError(res, 404, "Resource not found");
  });

  test("rejects access to another user's resource", async () => {
    const res = makeResponse();

    await ownerOrLawyer(async () => "user-2")(
      { user: { role: "client", _id: "user-1" } },
      res,
      jest.fn()
    );

    expectError(res, 403, "Access denied. You do not own this resource.");
  });

  test("allows access to the user's own resource", async () => {
    const next = jest.fn();

    await ownerOrLawyer(async () => ({ toString: () => "user-1" }))(
      {
        user: {
          role: "client",
          _id: { toString: () => "user-1" },
        },
      },
      makeResponse(),
      next
    );

    expect(next).toHaveBeenCalledTimes(1);
  });

  test("converts ownership lookup failures into authorization errors", async () => {
    const res = makeResponse();

    await ownerOrLawyer(async () => {
      throw new Error("database unavailable");
    })(
      { user: { role: "client", _id: "user-1" } },
      res,
      jest.fn()
    );

    expectError(res, 500, "Authorization check failed");
  });
});
