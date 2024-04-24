#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from locust import HttpUser, TaskSet, between, constant, task
from faker import Faker
import datetime

fake = Faker()

products = [
    '0PUK6V6EV0',
    '1YMWWN1N4O',
    '2ZYFJ3GM2N',
    '66VCHSJNUP',
    '6E92ZMYYFZ',
    '9SIQT8TOJO',
    'L9ECAV7KIM',
    'LS4PSXUNUM',
    'OLJCESPC7Z']


def index(l):
    l.client.get("/", name="home")


def setCurrency(l):
    currencies = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY']
    l.client.post("/setCurrency",
                  {'currency_code': random.choice(currencies)}, name="setCurrency")


def browseProduct(l):
    l.client.get("/product/" + random.choice(products), name="viewProduct")


def viewCart(l):
    l.client.get("/cart", name="viewCart")


def addToCart(l):
    product = random.choice(products)
    l.client.get("/product/" + product, name="viewProduct")
    l.client.post("/cart", {
        'product_id': product,
        'quantity': random.randint(1, 10)}, name="addToCart")


def empty_cart(l):
    l.client.post('/cart/empty', name="cartEmpty")


def checkout(l):
    addToCart(l)
    current_year = datetime.datetime.now().year + 1
    l.client.post("/cart/checkout", {
        'email': fake.email(),
        'street_address': fake.street_address(),
        'zip_code': fake.zipcode(),
        'city': fake.city(),
        'state': fake.state_abbr(),
        'country': fake.country(),
        'credit_card_number': fake.credit_card_number(card_type="visa"),
        'credit_card_expiration_month': random.randint(1, 12),
        'credit_card_expiration_year': random.randint(current_year, current_year + 70),
        'credit_card_cvv': f"{random.randint(100, 999)}",
    }, name="checkout")


def logout(l):
    l.client.get('/logout', name="logout")


def perform_operation(self, name):
    all_operations = {"index": index,
                      "setCurrency": setCurrency,
                      "browseProduct": browseProduct,
                      "viewCart": viewCart,
                      "addToCart": addToCart,
                      "empty_cart": empty_cart,
                      "checkout": checkout,
                      "logout": logout}
    operation = all_operations.get(name)
    operation(self)


class UserNormal(HttpUser):
    weight = 70
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["index", "setCurrency", "browseProduct", "index",
                      "browseProduct", "index", "browseProduct", "index",
                      "addToCart", "viewCart", "index", "browseProduct", "logout"]

        for operation in operations:
            perform_operation(self, operation)


class UserStressCart(HttpUser):
    weight = 15
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["index", "setCurrency", "browseProduct", "addToCart", "index",
                      "browseProduct", "addToCart", "index", "empty_cart", "checkout", "index",
                      "browseProduct", "addToCart", "index",
                      "browseProduct", "viewCart", "empty_cart", "index", "addToCart", "index", "browseProduct",
                      "empty_cart", "logout"]

        for operation in operations:
            perform_operation(self, operation)


class UserStressShop(HttpUser):
    weight = 15
    wait_time = constant(1)

    @task
    def perform_task(self):
        operations = ["index", "setCurrency", "browseProduct", "addToCart", "index",
                      "browseProduct", "addToCart", "index", "viewCart", "empty_cart", "index",
                      "browseProduct", "addToCart", "viewCart",
                      "checkout", "logout"]

        for operation in operations:
            perform_operation(self, operation)