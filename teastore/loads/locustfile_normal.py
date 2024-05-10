import logging
from random import randint, choice
from locust import HttpUser, task, constant


def visit_home(self):
    self.client.get('/', name="home")


def login(self):
    self.client.get('/login', name="login_page")
    user = randint(1, 99)
    self.client.post("/loginAction", params={"username": user, "password": "password"}, name="login")


def browse(self):
    category_id = randint(2, 6)
    page = randint(1, 5)
    self.client.get("/category", params={"page": page, "category": category_id}, name="category")


def view_product(self):
    product_id = randint(7, 506)
    self.client.get("/product", params={"id": product_id}, name="view_product")


def add_to_cart(self):
    product_id = randint(7, 506)
    product_request = self.client.get("/product", params={"id": product_id}, name="product")
    if product_request.ok:
        self.client.post("/cartAction", params={"addToCart": "", "productid": product_id}, name="addToCart")


def buy(self):
    # sample user data
    user_data = {
        "firstname": "User",
        "lastname": "User",
        "adress1": "Road",
        "adress2": "City",
        "cardtype": "volvo",
        "cardnumber": "314159265359",
        "expirydate": "12/2050",
        "confirm": "Confirm"
    }
    self.client.post("/cartAction", params=user_data, name="buy")


def visit_profile(self):
    self.client.get("/profile", name="profile")


def logout(self):
    self.client.post("/loginAction", params={"logout": ""}, name="logout")


def perform_operation(self, name):
    all_operations = {"visit_home": visit_home,
                      "login": login,
                      "browse": browse,
                      "view_product": view_product,
                      "add_to_cart": add_to_cart,
                      "buy": buy,
                      "visit_profile": visit_profile,
                      "logout": logout}
    operation = all_operations.get(name)
    if operation == None:
        print(name)

    operation(self)


class UserNormal(HttpUser):
    weight = 70
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["visit_home", "login", "visit_profile", "visit_home", "browse", "browse", "view_product",
                      "visit_home", "browse", "browse", "view_product", "visit_home", "browse", "visit_home",
                      "add_to_cart", "visit_home", "logout"]

        for operation in operations:
            perform_operation(self, operation)


class UserStressCart(HttpUser):
    weight = 15
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["visit_home", "login", "visit_profile", "visit_home", "browse", "browse", "view_product",
                      "add_to_cart", "visit_home", "browse", "browse", "view_product", "add_to_cart",
                      "visit_home", "browse", "view_product", "add_to_cart", "visit_home", "logout"]

        for operation in operations:
            perform_operation(self, operation)


class UserStressShop(HttpUser):
    weight = 15
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["visit_home", "login", "visit_profile", "visit_home", "browse", "browse", "view_product",
                      "add_to_cart", "visit_home", "browse", "browse", "buy", "visit_home", "logout"]

        for operation in operations:
            perform_operation(self, operation)
