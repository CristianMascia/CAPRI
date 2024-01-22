# This file describes the test execution. It is based on a previous version, developed for Faban (http://faban.org), which,
# for documentation reasons, is attached at the bottom of this file.

import random
import json
import datetime
import secrets
from locust import HttpUser, task, between, constant
import requests

import locust.stats
locust.stats.CONSOLE_STATS_INTERVAL_SEC = 1
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10
locust.stats.PERCENTILES_TO_REPORT = [0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.98, 0.99, 0.999, 0.9999, 1.0]


def get_home(self):
    self.client.get("/index.html", verify=False, name="get_index")


def get_login(self):
    head = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": "Basic dXNlcjpwYXNzd29yZA=="}
    self.client.get(url="/login", headers=head, verify=False, name="login")


def get_catalog_variant_1(self):
    self.client.get("/catalogue?size=5", verify=False, name="get_catalogue1")


def get_catalog_variant_2(self):
    self.client.get("/catalogue/size", verify=False, name="get_catalogue2")


def get_catalog_variant_3(self):
    self.client.get("/catalogue?page=1&size=6", verify=False, name="get_catalogue3")


def get_category(self):
    head = {"Content-Type": "html"}
    self.client.get(url="/category.html", headers=head, verify=False, name="get_category")


def get_item(self):
    self.client.get("/catalogue/3395a43e-2d88-40de-b95f-e00e1502085b", verify=False, name="get_item")


def get_related(self):
    self.client.get("/catalogue?sort=id&size=3&tags=brown", verify=False, name="get_related")


def get_details(self):
    head = {"Content-Type": "html"}
    self.client.get(url="/detail.html?id=3395a43e-2d88-40de-b95f-e00e1502085b", headers=head, verify=False, name="get_detail")


def get_tags(self):
    self.client.get("/tags", verify=False, name="get_tags")


def get_cart(self):
    self.client.get("/cart", verify=False, name="get_cart")


def add_to_cart(self):
    head = {"Content-Type": "application/json"}
    self.client.post(url="/cart", headers=head, json={"id": "3395a43e-2d88-40de-b95f-e00e1502085b"}, name="add_item_to_cart")


def get_basket(self):
    head = {"Content-Type": "application/json"}
    self.client.get(url="/basket.html", headers=head, verify=False, name="get_basket")


def get_orders(self):
    self.client.get("/orders", verify=False, name="get_orders")


def get_all_orders(self):
    self.client.get("/customer-orders.html", verify=False, name="get_customer_orders")


def get_customer(self):
    self.client.get("/customers/fz5cpW831_cB4MMSsuphqSgPw7XHYHa0", verify=False, name="get_customer")


def get_card(self):
    self.client.get("/card", verify=False, name="get_card")


def get_address(self):
    self.client.get("/address", verify=False, name="get_address")


def perform_operation(self, name):
    all_operations = {"home": get_home, "login": get_login, "getCatalogue": get_catalog_variant_1, "catalogueSize": get_catalog_variant_2, "cataloguePage": get_catalog_variant_3, "catalogue": get_category, "getItem": get_item, "getRelated": get_related, "showDetails": get_details, "tags": get_tags, "getCart": get_cart, "addToCart": add_to_cart, "basket": get_basket, "createOrder": get_orders, "getOrders": get_orders, "viewOrdersPage": get_all_orders, "getCustomer": get_customer, "getCard": get_card, "getAddress": get_address}
    operation = all_operations.get(name)
    operation(self)


class UserNoLogin(HttpUser):
    weight = 10
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["home", "getCatalogue", "getCart", "home", "getCatalogue", "getCart", "catalogue", "catalogueSize", "tags", "cataloguePage", "getCart", "getCustomer", "showDetails", "getItem", "getCustomer", "getCart", "getRelated"]

        for operation in operations:
            perform_operation(self, operation)


class UserLoginAndShop(HttpUser):
    weight = 20
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["home", "getCatalogue", "getCart", "login", "home", "getCatalogue", "getCart", "home", "getCatalogue", "getCart", "catalogue", "catalogueSize", "tags", "cataloguePage", "getCart", "getCustomer", "showDetails", "getItem", "getCustomer", "getCart", "getRelated", "addToCart", "showDetails", "getItem", "getCustomer", "getCart", "getRelated", "basket", "getCart", "getCard", "getAddress", "getCatalogue", "getItem", "getCart", "getCustomer", "getItem", "createOrder", "viewOrdersPage", "getOrders", "getCart", "getCustomer", "getItem"]

        for operation in operations:
            perform_operation(self, operation)


class UserLoginAndCheckCart(HttpUser):
    weight = 70
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["home", "getCatalogue", "getCart", "login", "home", "getCatalogue", "getCart", "viewOrdersPage", "getOrders", "getCart", "getCustomer", "getItem"]

        for operation in operations:
            perform_operation(self, operation)


