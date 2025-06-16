# Brain-Tumour-Detector

# 🛒 Flipkart-Style E-Commerce Cart & Checkout API

This is a RESTful API backend project that simulates key functionalities of a Flipkart-style e-commerce platform, including user authentication, product listings, cart operations, and order checkout. Built using **Flask** and **PostgreSQL**, this project showcases backend engineering and clean API architecture for a scalable e-commerce system.

---

## 🚀 Tech Stack

- **Backend:** Python, Flask
- **Database:** PostgreSQL (or SQLite for local dev)
- **ORM:** SQLAlchemy
- **Authentication:** JWT (JSON Web Tokens)
- **Tools:** Postman, Flask-Migrate, Swagger UI (optional)

---

## 📦 Features

### 👤 User Module
- User registration and login
- JWT-based authentication
- View and update profile

### 🛍️ Product Module
- Add, update, delete products (Admin only)
- View product listings with filtering (category, price, search)
- Check product availability and stock

### 🛒 Cart Module
- Add to cart (with quantity)
- View cart contents
- Update item quantities
- Remove items
- Auto-calculate total with tax/discount (basic)

### 💳 Checkout Module
- Checkout cart
- Simulate payment status (success/failure)
- View past orders

### 📊 (Optional) Analytics
- Most viewed products
- Cart abandonment stats
- Order frequency by user

---

## 🗂️ Project Structure


---

## 📄 Sample API Endpoints

| Method | Endpoint               | Description                  |
|--------|------------------------|------------------------------|
| POST   | `/api/register`        | Register new user            |
| POST   | `/api/login`           | Login and get JWT token      |
| GET    | `/api/products`        | List all products            |
| POST   | `/api/cart`            | Add product to cart          |
| GET    | `/api/cart`            | View current cart            |
| POST   | `/api/checkout`        | Checkout cart (mock payment) |
| GET    | `/api/orders`          | View past orders             |

---

## 📦 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ecommerce-api.git
cd ecommerce-api

# 2. Set up virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (DB URI, JWT secret, etc.)
# Example for SQLite in config.py or .env

# 5. Run migrations (if using Flask-Migrate)
flask db init
flask db migrate
flask db upgrade

# 6. Run the app
python main.py
