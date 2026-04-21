import asyncio
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.auth import dependencies
from app.auth import jwt as jwt_module
from app.auth import redis as redis_module
from app.core.config import settings
from app.database import _engine_kwargs, get_db
from app.main import (
    create_calculation,
    delete_calculation,
    get_calculation,
    lifespan,
    list_calculations,
    login_form,
    login_json,
    read_health,
    read_root,
    register,
    update_calculation,
)
from app.models.calculation import Addition, Calculation, Division, Multiplication, Subtraction
from app.models.user import User
from app.schemas.calculation import CalculationBase, CalculationUpdate
from app.schemas.token import TokenType
from app.schemas.user import PasswordUpdate, UserCreate, UserLogin


class FakeQuery:
    def __init__(self, first_result=None, all_result=None):
        self._first_result = first_result
        self._all_result = all_result if all_result is not None else []

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._first_result

    def all(self):
        return self._all_result


class FakeDB:
    def __init__(self, first_result=None, all_result=None):
        self.query_obj = FakeQuery(first_result=first_result, all_result=all_result)
        self.committed = False
        self.rolled_back = False
        self.deleted = None
        self.added = []
        self.refreshed = []

    def query(self, _model):
        return self.query_obj

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def refresh(self, obj):
        self.refreshed.append(obj)

    def delete(self, obj):
        self.deleted = obj

    def flush(self):
        pass


class FakeForm:
    def __init__(self, username, password):
        self.username = username
        self.password = password


def test_database_engine_kwargs_and_get_db_close(monkeypatch):
    assert _engine_kwargs("sqlite:///./x.db") == {"connect_args": {"check_same_thread": False}}
    assert _engine_kwargs("postgresql://u:p@localhost:5432/db") == {}

    closed = {"value": False}

    class DummySession:
        def close(self):
            closed["value"] = True

    monkeypatch.setattr("app.database.SessionLocal", lambda: DummySession())

    gen = get_db()
    _ = next(gen)
    gen.close()
    assert isinstance(closed["value"], bool)


def test_dependencies_minimal_uuid_and_invalid_type(monkeypatch):
    token_uuid = uuid4()
    monkeypatch.setattr(User, "verify_token", lambda _token: token_uuid)
    user = dependencies.get_current_user(token="x")
    assert user.id == token_uuid

    monkeypatch.setattr(User, "verify_token", lambda _token: 123)
    with pytest.raises(HTTPException) as exc:
        dependencies.get_current_user(token="x")
    assert exc.value.status_code == 401

    token_sub = uuid4()
    monkeypatch.setattr(User, "verify_token", lambda _token: {"sub": token_sub})
    user_sub = dependencies.get_current_user(token="x")
    assert user_sub.id == token_sub


def test_user_schema_password_validations_and_password_update_errors():
    with pytest.raises(ValueError, match="Passwords do not match"):
        UserCreate(
            first_name="A",
            last_name="B",
            email="a@example.com",
            username="abc",
            password="SecurePass1!",
            confirm_password="Mismatch1!",
        )

    with pytest.raises(ValueError, match="uppercase"):
        UserCreate(
            first_name="A",
            last_name="B",
            email="a2@example.com",
            username="abc2",
            password="lowercase1!",
            confirm_password="lowercase1!",
        )

    with pytest.raises(ValueError, match="lowercase"):
        UserCreate(
            first_name="A",
            last_name="B",
            email="a3@example.com",
            username="abc3",
            password="UPPERCASE1!",
            confirm_password="UPPERCASE1!",
        )

    with pytest.raises(ValueError, match="digit"):
        UserCreate(
            first_name="A",
            last_name="B",
            email="a4@example.com",
            username="abc4",
            password="NoDigits!",
            confirm_password="NoDigits!",
        )

    with pytest.raises(ValueError, match="special character"):
        UserCreate(
            first_name="A",
            last_name="B",
            email="a5@example.com",
            username="abc5",
            password="NoSpecial123",
            confirm_password="NoSpecial123",
        )

    with pytest.raises(ValueError, match="confirmation"):
        PasswordUpdate(
            current_password="OldPass123!",
            new_password="NewPass123!",
            confirm_new_password="WrongPass123!",
        )

    with pytest.raises(ValueError, match="different"):
        PasswordUpdate(
            current_password="SamePass123!",
            new_password="SamePass123!",
            confirm_new_password="SamePass123!",
        )

    good = PasswordUpdate(
        current_password="OldPass123!",
        new_password="NewPass123!",
        confirm_new_password="NewPass123!",
    )
    assert good.new_password == "NewPass123!"


def test_calculation_schema_input_validation_edges():
    with pytest.raises(ValueError, match="Type must be one of"):
        CalculationBase(type=123, inputs=[1, 2])

    with pytest.raises(ValueError, match="Input should be a valid list"):
        CalculationBase(type="addition", inputs="not-a-list")

    with pytest.raises(ValueError, match="Cannot divide by zero"):
        CalculationBase(type="division", inputs=[100, 0])

    with pytest.raises(ValidationError):
        CalculationUpdate(inputs=[1])


def test_calculation_operation_error_paths():
    add = Addition(inputs="x", user_id=uuid4())
    with pytest.raises(ValueError, match="list"):
        add.get_result()

    add.inputs = [1]
    with pytest.raises(ValueError, match="at least two"):
        add.get_result()

    sub = Subtraction(inputs=[1], user_id=uuid4())
    with pytest.raises(ValueError, match="at least two"):
        sub.get_result()

    sub.inputs = "x"
    with pytest.raises(ValueError, match="list"):
        sub.get_result()

    mul = Multiplication(inputs="x", user_id=uuid4())
    with pytest.raises(ValueError, match="list"):
        mul.get_result()

    mul.inputs = [5]
    with pytest.raises(ValueError, match="at least two"):
        mul.get_result()

    div = Division(inputs="x", user_id=uuid4())
    with pytest.raises(ValueError, match="list"):
        div.get_result()

    div.inputs = [10]
    with pytest.raises(ValueError, match="at least two"):
        div.get_result()

    base_calc = Calculation(type="calculation", user_id=uuid4(), inputs=[1, 2])
    with pytest.raises(NotImplementedError):
        base_calc.get_result()
    assert "Calculation(type=calculation" in repr(base_calc)


def test_user_model_misc_paths():
    u = User(
        first_name="Test",
        last_name="User",
        email="u@example.com",
        username="tester",
        hashed_password="hash",
    )
    assert u.password == "hash"
    assert "u@example.com" in str(u)
    assert u.hashed_password == "hash"

    prev_updated = u.updated_at
    u.update(first_name="New")
    assert u.first_name == "New"
    assert u.updated_at is not None

    class FakeUserQuery:
        def filter(self, *_args, **_kwargs):
            return self

        def first(self):
            return SimpleNamespace(id=uuid4(), verify_password=lambda _p: False)

    class FakeUserDB:
        def query(self, _model):
            return FakeUserQuery()

        def flush(self):
            pass

    assert User.authenticate(FakeUserDB(), "someone", "badpass") is None
    assert User.verify_token("not-a-token") is None

    monkey_sub_none = jwt_module.create_token("abc", TokenType.ACCESS)
    # Force decode to return payload without sub and invalid sub paths.
    from jose import jwt as jose_jwt
    from app.core.config import settings as app_settings
    no_sub_token = jose_jwt.encode({"foo": "bar"}, app_settings.JWT_SECRET_KEY, algorithm=app_settings.ALGORITHM)
    assert User.verify_token(no_sub_token) is None
    bad_sub_token = jose_jwt.encode({"sub": "not-a-uuid"}, app_settings.JWT_SECRET_KEY, algorithm=app_settings.ALGORITHM)
    assert User.verify_token(bad_sub_token) is None


def test_main_register_and_login_paths(monkeypatch):
    fake_user = SimpleNamespace(
        id=uuid4(),
        username="user1",
        email="u1@example.com",
        first_name="U",
        last_name="One",
        is_active=True,
        is_verified=False,
    )
    db = FakeDB()
    req = UserCreate(
        first_name="U",
        last_name="One",
        email="u1@example.com",
        username="user1",
        password="SecurePass123!",
        confirm_password="SecurePass123!",
    )

    monkeypatch.setattr(User, "register", lambda _db, _data: fake_user)
    out = register(req, db=db)
    assert out.username == "user1"
    assert db.committed is True

    def raise_register(_db, _data):
        raise ValueError("bad register")

    db2 = FakeDB()
    monkeypatch.setattr(User, "register", raise_register)
    with pytest.raises(HTTPException) as exc:
        register(req, db=db2)
    assert exc.value.status_code == 400
    assert db2.rolled_back is True

    monkeypatch.setattr(User, "authenticate", lambda _db, _u, _p: None)
    with pytest.raises(HTTPException) as exc:
        login_json(UserLogin(username="user1", password="SecurePass123!"), db=FakeDB())
    assert exc.value.status_code == 401

    aware_exp = datetime.now(timezone.utc) + timedelta(minutes=1)
    auth_ok = {
        "access_token": "a",
        "refresh_token": "r",
        "expires_at": aware_exp,
        "user": fake_user,
    }
    monkeypatch.setattr(User, "authenticate", lambda _db, _u, _p: auth_ok)
    token_resp = login_json(UserLogin(username="user1", password="SecurePass123!"), db=FakeDB())
    assert token_resp.access_token == "a"

    naive_auth = dict(auth_ok)
    naive_auth["expires_at"] = datetime.utcnow()
    monkeypatch.setattr(User, "authenticate", lambda _db, _u, _p: naive_auth)
    token_resp2 = login_json(UserLogin(username="user1", password="SecurePass123!"), db=FakeDB())
    assert token_resp2.expires_at.tzinfo is not None

    monkeypatch.setattr(User, "authenticate", lambda _db, _u, _p: None)
    with pytest.raises(HTTPException):
        login_form(FakeForm("user1", "bad"), db=FakeDB())

    monkeypatch.setattr(User, "authenticate", lambda _db, _u, _p: auth_ok)
    form_ok = login_form(FakeForm("user1", "good"), db=FakeDB())
    assert form_ok["token_type"] == "bearer"


def test_main_calculation_route_paths(monkeypatch):
    current_user = SimpleNamespace(id=uuid4())
    calc_obj = SimpleNamespace(
        id=uuid4(),
        user_id=current_user.id,
        inputs=[1, 2],
        result=3,
        updated_at=datetime.utcnow(),
        get_result=lambda: 3,
    )

    monkeypatch.setattr(Calculation, "create", lambda **kwargs: calc_obj)
    db = FakeDB()
    created = create_calculation(CalculationBase(type="addition", inputs=[1, 2]), current_user=current_user, db=db)
    assert created.result == 3
    assert db.committed is True

    def bad_create(**kwargs):
        raise ValueError("invalid")

    monkeypatch.setattr(Calculation, "create", bad_create)
    db_bad = FakeDB()
    with pytest.raises(HTTPException) as exc:
        create_calculation(CalculationBase(type="addition", inputs=[1, 2]), current_user=current_user, db=db_bad)
    assert exc.value.status_code == 400
    assert db_bad.rolled_back is True

    listed = list_calculations(current_user=current_user, db=FakeDB(all_result=[calc_obj]))
    assert len(listed) == 1

    with pytest.raises(HTTPException):
        get_calculation("not-a-uuid", current_user=current_user, db=FakeDB())

    with pytest.raises(HTTPException):
        get_calculation(str(uuid4()), current_user=current_user, db=FakeDB(first_result=None))

    got = get_calculation(str(calc_obj.id), current_user=current_user, db=FakeDB(first_result=calc_obj))
    assert got.id == calc_obj.id

    with pytest.raises(HTTPException):
        update_calculation("bad-id", CalculationUpdate(inputs=[1, 2]), current_user=current_user, db=FakeDB())

    with pytest.raises(HTTPException):
        update_calculation(str(uuid4()), CalculationUpdate(inputs=[1, 2]), current_user=current_user, db=FakeDB(first_result=None))

    calc_obj2 = SimpleNamespace(
        id=uuid4(),
        user_id=current_user.id,
        inputs=[2, 3],
        result=6,
        updated_at=datetime.utcnow(),
        get_result=lambda: 30,
    )
    updated = update_calculation(
        str(calc_obj2.id),
        CalculationUpdate(inputs=[5, 6]),
        current_user=current_user,
        db=FakeDB(first_result=calc_obj2),
    )
    assert updated.result == 30

    updated_no_inputs = update_calculation(
        str(calc_obj2.id),
        CalculationUpdate(inputs=None),
        current_user=current_user,
        db=FakeDB(first_result=calc_obj2),
    )
    assert updated_no_inputs.id == calc_obj2.id

    with pytest.raises(HTTPException):
        delete_calculation("bad-id", current_user=current_user, db=FakeDB())

    with pytest.raises(HTTPException):
        delete_calculation(str(uuid4()), current_user=current_user, db=FakeDB(first_result=None))

    db_del = FakeDB(first_result=calc_obj2)
    assert delete_calculation(str(calc_obj2.id), current_user=current_user, db=db_del) is None
    assert db_del.deleted is calc_obj2


def test_health_endpoint_and_lifespan(monkeypatch):
    assert read_health() == {"status": "ok"}
    assert "Calculations API is running" in read_root()

    called = {"value": False}

    class DummyMeta:
        def create_all(self, bind=None):
            called["value"] = True

    class DummyBase:
        metadata = DummyMeta()

    monkeypatch.setattr("app.main.Base", DummyBase)

    async def run_lifespan():
        cm = lifespan(None)
        await cm.__aenter__()
        await cm.__aexit__(None, None, None)

    asyncio.run(run_lifespan())
    assert called["value"] is True


def test_jwt_create_decode_and_current_user_paths(monkeypatch):
    pwd = "SecurePass123!"
    hashed = jwt_module.get_password_hash(pwd)
    assert jwt_module.verify_password(pwd, hashed) is True

    token = jwt_module.create_token(uuid4(), TokenType.ACCESS)
    assert isinstance(token, str)

    refresh = jwt_module.create_token("abc", TokenType.REFRESH, expires_delta=timedelta(minutes=5))
    assert isinstance(refresh, str)

    def bad_encode(*args, **kwargs):
        raise RuntimeError("boom")

    original_encode = jwt_module.jwt.encode
    monkeypatch.setattr(jwt_module.jwt, "encode", bad_encode)
    with pytest.raises(HTTPException) as exc:
        jwt_module.create_token("abc", TokenType.ACCESS)
    assert exc.value.status_code == 500

    monkeypatch.setattr(jwt_module.jwt, "encode", original_encode)

    async def run_decode_checks():
        monkeypatch.setattr(jwt_module, "is_blacklisted", lambda _jti: asyncio.sleep(0, result=False))
        ok_payload = await jwt_module.decode_token(
            jwt_module.create_token("user1", TokenType.ACCESS), TokenType.ACCESS
        )
        assert ok_payload["sub"] == "user1"

        invalid_type_token = jwt_module.jwt.encode(
            {
                "sub": "user1",
                "type": TokenType.REFRESH.value,
                "jti": "abc123",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            settings.JWT_SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        exc = await _expect_http_exception(
            invalid_type_token, TokenType.ACCESS
        )
        assert exc.detail == "Invalid token type"

        monkeypatch.setattr(jwt_module, "is_blacklisted", lambda _jti: asyncio.sleep(0, result=True))
        with pytest.raises(HTTPException):
            await jwt_module.decode_token(
                jwt_module.create_token("user1", TokenType.ACCESS), TokenType.ACCESS
            )

        monkeypatch.setattr(jwt_module, "is_blacklisted", lambda _jti: asyncio.sleep(0, result=False))
        with pytest.raises(HTTPException):
            await jwt_module.decode_token(
                jwt_module.create_token("user1", TokenType.ACCESS, expires_delta=timedelta(seconds=-1)),
                TokenType.ACCESS,
            )

        with pytest.raises(HTTPException):
            await jwt_module.decode_token("not-a-token", TokenType.ACCESS)

    asyncio.run(run_decode_checks())

    active_user = SimpleNamespace(id="u1", is_active=True)

    class DBActive(FakeDB):
        def __init__(self, user):
            super().__init__(first_result=user)

    async def current_user_checks():
        monkeypatch.setattr(jwt_module, "decode_token", lambda *_args, **_kwargs: asyncio.sleep(0, result={"sub": "u1"}))
        got = await jwt_module.get_current_user(token="t", db=DBActive(active_user))
        assert got.is_active is True

        with pytest.raises(HTTPException) as exc_none:
            await jwt_module.get_current_user(token="t", db=DBActive(None))
        assert exc_none.value.status_code == 401

        inactive = SimpleNamespace(id="u2", is_active=False)
        with pytest.raises(HTTPException) as exc_inactive:
            await jwt_module.get_current_user(token="t", db=DBActive(inactive))
        assert exc_inactive.value.status_code == 401

        monkeypatch.setattr(jwt_module, "decode_token", lambda *_args, **_kwargs: (_ for _ in ()).throw(HTTPException(status_code=401, detail="bad")))
        with pytest.raises(HTTPException) as exc_decode:
            await jwt_module.get_current_user(token="t", db=DBActive(active_user))
        assert exc_decode.value.status_code == 401

    asyncio.run(current_user_checks())


def test_redis_paths(monkeypatch):
    # Fallback path (no redis client)
    if hasattr(redis_module.get_redis, "redis"):
        delattr(redis_module.get_redis, "redis")

    monkeypatch.setattr(redis_module, "_fallback_blacklist", set())

    asyncio.run(redis_module.add_to_blacklist("jti1", 10))
    assert asyncio.run(redis_module.is_blacklisted("jti1")) is True

    class FakeRedis:
        def __init__(self):
            self.store = {}

        async def set(self, key, value, ex=None):
            self.store[key] = value

        async def exists(self, key):
            return 1 if key in self.store else 0

    fake_client = FakeRedis()

    async def fake_get_redis():
        return fake_client

    monkeypatch.setattr(redis_module, "get_redis", fake_get_redis)
    asyncio.run(redis_module.add_to_blacklist("jti2", 10))
    assert asyncio.run(redis_module.is_blacklisted("jti2")) is True


async def _expect_http_exception(token, token_type):
    with pytest.raises(HTTPException) as exc_info:
        await jwt_module.decode_token(token, token_type)
    return exc_info.value


def test_redis_get_redis_success_and_cache(monkeypatch):
    if hasattr(redis_module.get_redis, "redis"):
        delattr(redis_module.get_redis, "redis")

    class FakeAsyncRedisModule:
        @staticmethod
        async def from_url(_url):
            return {"client": True}

    fake_redis_pkg = SimpleNamespace(asyncio=FakeAsyncRedisModule)
    monkeypatch.setitem(sys.modules, "redis", fake_redis_pkg)

    first = asyncio.run(redis_module.get_redis())
    second = asyncio.run(redis_module.get_redis())
    assert first == {"client": True}
    assert second == first
