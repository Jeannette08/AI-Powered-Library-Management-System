import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import mysql.connector
from datetime import datetime, timedelta
import hashlib
import re
import os
import shutil
from PIL import Image, ImageTk
import qrcode
import random
import numpy as np
from PIL import Image, ImageTk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



class GeneticRecommendationEngine:
    """Genetic Algorithm for personalized book recommendations"""
    
    def __init__(self, population_size=20, generations=30, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def generate_recommendations(self, cursor, user_id, num_recommendations=5):
        """
        Generate book recommendations using Genetic Algorithm
        
        Fitness Function considers:
        - User's borrowing history (categories preferred)
        - Similar users' borrowing patterns
        - Book popularity
        - Book availability
        """
        
        # Get user's borrowing history
        cursor.execute("""
            SELECT DISTINCT b.category
            FROM transactions t
            JOIN books b ON t.book_id = b.book_id
            WHERE t.user_id = %s AND b.category IS NOT NULL
        """, (user_id,))
        
        user_categories = [row[0] for row in cursor.fetchall()]
        
        # Get all available books not currently borrowed by user
        cursor.execute("""
            SELECT b.book_id, b.title, b.author, b.category, b.available,
                COUNT(t.transaction_id) as borrow_count
            FROM books b
            LEFT JOIN transactions t ON b.book_id = t.book_id
            WHERE b.book_id NOT IN (
                SELECT book_id FROM transactions WHERE user_id = %s AND status = 'borrowed'
            )
            AND b.available > 0
            GROUP BY b.book_id
        """, (user_id,))
        
        available_books = cursor.fetchall()
        
        if not available_books:
            return []
        
        # Get similar users (users who borrowed same categories)
        cursor.execute("""
            SELECT DISTINCT t2.user_id
            FROM transactions t1
            JOIN books b1 ON t1.book_id = b1.book_id
            JOIN transactions t2 ON t2.book_id IN (
                SELECT book_id FROM books WHERE category = b1.category
            )
            WHERE t1.user_id = %s AND t2.user_id != %s
            LIMIT 10
        """, (user_id, user_id))
        
        similar_users = [row[0] for row in cursor.fetchall()]
        
        # Get books borrowed by similar users
        similar_user_books = set()
        if similar_users:
            placeholders = ','.join(['%s'] * len(similar_users))
            cursor.execute(f"""
                SELECT DISTINCT book_id
                FROM transactions
                WHERE user_id IN ({placeholders})
            """, similar_users)
            similar_user_books = {row[0] for row in cursor.fetchall()}
        
        # Initialize population (random book combinations)
        population = self._initialize_population(available_books)
        
        # Evolve population
        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitness_scores = [
                self._calculate_fitness(
                    individual, available_books, user_categories, 
                    similar_user_books
                )
                for individual in population
            ]
            
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(parents)
            
            # Mutation
            offspring = self._mutation(offspring, len(available_books))
            
            # New generation
            population = offspring
        
        # Get best individual from final population
        final_fitness = [
            self._calculate_fitness(
                individual, available_books, user_categories, 
                similar_user_books
            )
            for individual in population
        ]
        
        best_individual = population[final_fitness.index(max(final_fitness))]
        
        # Convert indices to book data
        recommendations = []
        for idx in best_individual[:num_recommendations]:
            if idx < len(available_books):
                book_data = available_books[idx]
                recommendations.append({
                    'book_id': book_data[0],
                    'title': book_data[1],
                    'author': book_data[2],
                    'category': book_data[3],
                    'available': book_data[4],
                    'popularity': book_data[5]
                })
        
        return recommendations
    
    def _initialize_population(self, available_books):
        """Create initial random population"""
        population = []
        book_count = len(available_books)
        
        for _ in range(self.population_size):
            # Each individual is a list of book indices
            individual = random.sample(range(book_count), min(10, book_count))
            population.append(individual)
        
        return population
    
    def _calculate_fitness(self, individual, available_books, user_categories, similar_user_books):
        """
        Calculate fitness score for an individual
        
        Factors:
        - Category match (40%)
        - Similar user preference (30%)
        - Popularity (20%)
        - Diversity (10%)
        """
        if not individual:
            return 0
        
        fitness = 0
        categories_in_individual = []
        
        for idx in individual:
            if idx >= len(available_books):
                continue
            
            book_data = available_books[idx]
            book_id, title, author, category, available, popularity = book_data
            
            # Category match score (40 points max)
            if category in user_categories:
                fitness += 40
            
            categories_in_individual.append(category)
            
            # Similar user preference (30 points max)
            if book_id in similar_user_books:
                fitness += 30
            
            # Popularity score (20 points max)
            # Normalize popularity (max 20 points)
            popularity_score = min(popularity * 2, 20)
            fitness += popularity_score
        
        # Diversity bonus (10 points max)
        # Reward having different categories
        unique_categories = len(set(categories_in_individual))
        diversity_score = min(unique_categories * 2, 10)
        fitness += diversity_score
        
        return fitness / len(individual)  # Average fitness per book
    
    def _selection(self, population, fitness_scores):
        """Tournament selection"""
        parents = []
        
        for _ in range(self.population_size):
            # Tournament with 3 individuals
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        
        return parents
    
    def _crossover(self, parents):
        """Single-point crossover"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            # FIX: Check if parents have enough genes for crossover
            if len(parent1) < 2 or len(parent2) < 2:
                # Just copy parents if too short
                offspring.extend([parent1[:], parent2[:]])
                continue
            
            if random.random() > 0.5:
                # Crossover point
                point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                child1 = parent1[:point] + [x for x in parent2[point:] if x not in parent1[:point]]
                child2 = parent2[:point] + [x for x in parent1[point:] if x not in parent2[:point]]
            else:
                child1, child2 = parent1[:], parent2[:]
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _mutation(self, offspring, book_count):
        """Random mutation"""
        for individual in offspring:
            if random.random() < self.mutation_rate:
                # Random mutation: swap or replace a gene
                if len(individual) > 1:
                    if random.random() > 0.5:
                        # Swap two positions
                        idx1, idx2 = random.sample(range(len(individual)), 2)
                        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
                    else:
                        # Replace with random book
                        idx = random.randint(0, len(individual) - 1)
                        individual[idx] = random.randint(0, book_count - 1)
        
        return offspring


class EmailService:
    """Email notification service using Gmail SMTP"""
    
    def __init__(self):
        # CONFIGURE YOUR GMAIL HERE:
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "jeannettejeannie08@gmail.com"  # CHANGE THIS
        self.sender_password = "toot vvpt mcua bbuc"   # CHANGE THIS (16-char from Gmail)
        self.sender_name = "LibraAI Library System"
    
    def send_email(self, recipient_email, subject, body_html):
        """Send email via Gmail SMTP"""
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.sender_name} <{self.sender_email}>"
            message["To"] = recipient_email
            
            # HTML body
            html_part = MIMEText(body_html, "html")
            message.attach(html_part)
            
            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            return True, "Email sent successfully"
        
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
    
    def create_html_template(self, title, content, footer=""):
        """Create HTML email template"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #e94560; color: white; padding: 20px; text-align: center; }}
                .content {{ background: #f4f4f4; padding: 30px; }}
                .footer {{ text-align: center; padding: 20px; color: #777; font-size: 12px; }}
                .button {{ display: inline-block; padding: 12px 30px; background: #27ae60; 
                          color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .warning {{ background: #fff3cd; border-left: 4px solid #f39c12; padding: 15px; margin: 20px 0; }}
                .info {{ background: #d1ecf1; border-left: 4px solid #0c5460; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📚 LibraAI</h1>
                    <p>AI-Powered Library Management System</p>
                </div>
                <div class="content">
                    <h2>{title}</h2>
                    {content}
                </div>
                <div class="footer">
                    {footer if footer else "This is an automated message from LibraAI Library System.<br>Please do not reply to this email."}
                </div>
            </div>
        </body>
        </html>
        """
    # ADD THIS METHOD:
    def send_password_reset_email(self, recipient_email, user_name, reset_code):
        """Send password reset code email"""
        subject = "🔐 Password Reset Code - LibraAI"
        
        content = f"""
        <p>Dear <strong>{user_name}</strong>,</p>
        
        <p>We received a request to reset your password for your LibraAI account.</p>
        
        <div class="warning">
            <p><strong>🔑 Your Password Reset Code:</strong></p>
            <h1 style="text-align: center; font-size: 48px; color: #e94560; letter-spacing: 5px;">
                {reset_code}
            </h1>
        </div>
        
        <p><strong>Important:</strong></p>
        <ul>
            <li>⏰ This code expires in 15 minutes</li>
            <li>🔒 Do not share this code with anyone</li>
            <li>❌ If you didn't request this, please ignore this email</li>
        </ul>
        
        <p>Enter this code in the password reset form to create a new password.</p>
        """
        
        html = self.create_html_template("Password Reset Request", content)
        
        return self.send_email(recipient_email, subject, html)

class LibraAISystem:
    def __init__(self, root):
        self.root = root
        self.root.title("LibraAI - AI-Powered Library Management System")
        self.root.geometry("1400x800")
        self.root.configure(bg="#1a1a2e")
        
        self.current_user = None
        self.user_role = None
        self.selected_image_path = None
        
        # Create images directory
        if not os.path.exists("book_images"):
            os.makedirs("book_images")
        if not os.path.exists("qr_codes"):
            os.makedirs("qr_codes")
        
        # Initialize database connection
        self.init_database()

        # ADD THIS LINE HERE:
        self.email_service = EmailService()
        
        # Show login screen
        self.show_login()
    
    def init_database(self):
        """Initialize database connection and create tables"""
        try:
            self.db = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="libraai_db"
            )
            self.cursor = self.db.cursor()
            self.create_tables()
        except mysql.connector.Error as err:
            if err.errno == 1049:
                temp_db = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password=""
                )
                temp_cursor = temp_db.cursor()
                temp_cursor.execute("CREATE DATABASE libraai_db")
                temp_cursor.close()
                temp_db.close()
                
                self.db = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="libraai_db"
                )
                self.cursor = self.db.cursor()
                self.create_tables()
            else:
                messagebox.showerror("Database Error", f"Error: {err}")
    
    def create_tables(self):
        """Create database tables"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                email VARCHAR(100),
                phone VARCHAR(20),
                role ENUM('student', 'librarian') NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add phone column if it doesn't exist (for existing databases)
        try:
            self.cursor.execute("SELECT phone FROM users LIMIT 1")
            self.cursor.fetchall()  # Consume the result
        except:
            try:
                self.cursor.execute("ALTER TABLE users ADD COLUMN phone VARCHAR(20) AFTER email")
                self.db.commit()
            except:
                pass
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                book_id INT AUTO_INCREMENT PRIMARY KEY,
                isbn VARCHAR(20) UNIQUE,
                title VARCHAR(200) NOT NULL,
                author VARCHAR(100) NOT NULL,
                category VARCHAR(50),
                quantity INT DEFAULT 1,
                available INT DEFAULT 1,
                location VARCHAR(50),
                image_path VARCHAR(255),
                qr_code_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add image_path column if it doesn't exist (for existing databases)
        try:
            self.cursor.execute("SELECT image_path FROM books LIMIT 1")
            self.cursor.fetchall()  # Consume the result
        except:
            try:
                self.cursor.execute("ALTER TABLE books ADD COLUMN image_path VARCHAR(255) AFTER location")
                self.db.commit()
            except:
                pass
            
        # Add qr_code_path column if it doesn't exist (for existing databases)
        try:
            self.cursor.execute("SELECT qr_code_path FROM books LIMIT 1")
            self.cursor.fetchall()  # Consume the result
        except:
            try:
                self.cursor.execute("ALTER TABLE books ADD COLUMN qr_code_path VARCHAR(255) AFTER image_path")
                self.db.commit()
            except:
                pass
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                book_id INT NOT NULL,
                borrow_date DATE NOT NULL,
                due_date DATE NOT NULL,
                return_date DATE,
                status ENUM('borrowed', 'returned', 'overdue') DEFAULT 'borrowed',
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (book_id) REFERENCES books(book_id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS penalties (
                penalty_id INT AUTO_INCREMENT PRIMARY KEY,
                transaction_id INT NOT NULL,
                user_id INT NOT NULL,
                amount DECIMAL(10, 2) NOT NULL,
                days_overdue INT NOT NULL,
                paid BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        try:
            self.cursor.execute("SELECT payment_date FROM penalties LIMIT 1")
            self.cursor.fetchall()
        except:
            try:
                self.cursor.execute("ALTER TABLE penalties ADD COLUMN payment_date DATE AFTER paid")
                self.db.commit()
            except:
                pass
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INT AUTO_INCREMENT PRIMARY KEY,
                category_name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by INT,
                FOREIGN KEY (created_by) REFERENCES users(user_id)
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                title VARCHAR(200) NOT NULL,
                message TEXT NOT NULL,
                type ENUM('due_soon', 'overdue', 'penalty', 'returned', 'info') DEFAULT 'info',
                is_read BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                book_id INT NOT NULL,
                status ENUM('pending', 'available', 'cancelled', 'fulfilled') DEFAULT 'pending',
                reserved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notified_at TIMESTAMP NULL,
                expires_at TIMESTAMP NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (book_id) REFERENCES books(book_id)
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                reset_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                reset_code VARCHAR(6) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        self.db.commit()
        
        self.cursor.execute("SELECT COUNT(*) FROM categories")
        if self.cursor.fetchone()[0] == 0:
            default_categories = [
                ("Fiction", "Novels, short stories, and imaginative literature"),
                ("Science", "Scientific research, textbooks, and discoveries"),
                ("History", "Historical events, biographies, and timelines"),
                ("Technology", "Computing, engineering, and technical subjects"),
                ("Business", "Management, finance, and entrepreneurship"),
                ("Arts", "Visual arts, music, theater, and creative expression"),
                ("Education", "Teaching methods, learning theories, and pedagogy"),
                ("Health", "Medicine, wellness, and healthcare"),
                ("Literature", "Classic and contemporary literary works"),
                ("Biography", "Life stories and memoirs"),
                ("Religion", "Religious texts and spiritual studies"),
                ("Philosophy", "Philosophical thoughts and theories"),
                ("Children", "Books for children and young readers"),
                ("Reference", "Dictionaries, encyclopedias, and reference materials"),
                ("Other", "Miscellaneous and uncategorized books")
            ]
            
            for cat_name, cat_desc in default_categories:
                self.cursor.execute("""
                    INSERT INTO categories (category_name, description)
                    VALUES (%s, %s)
                """, (cat_name, cat_desc))
            
            self.db.commit()

        # Insert default admin
        self.cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if not self.cursor.fetchone():
            hashed_pw = hashlib.sha256("Admin@123".encode()).hexdigest()
            self.cursor.execute("""
                INSERT INTO users (username, password, full_name, email, phone, role)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, ("admin", hashed_pw, "System Administrator", "admin@nilai.edu.my", "0123456789", "librarian"))
            self.db.commit()
    
    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_phone(self, phone):
        """Validate phone number (Malaysian format)"""
        pattern = r'^01[0-9]{8,9}$'
        return re.match(pattern, phone) is not None
    
    def validate_password(self, password):
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one number"
        if not re.search(r'[@$!%*?&#]', password):
            return False, "Password must contain at least one special character (@$!%*?&#)"
        return True, "Password is strong"
    
    def validate_username(self, username):
        """Validate username (alphanumeric only)"""
        pattern = r'^[a-zA-Z0-9_]{4,20}$'
        return re.match(pattern, username) is not None
    
    def validate_name(self, name):
        """Validate name (letters and spaces only)"""
        pattern = r'^[a-zA-Z\s]{2,100}$'
        return re.match(pattern, name) is not None
    
    def validate_isbn(self, isbn):
        """Validate ISBN format"""
        isbn_clean = isbn.replace('-', '').replace(' ', '')
        pattern = r'^(?:\d{10}|\d{13})$'
        return re.match(pattern, isbn_clean) is not None
    
    def validate_number(self, value):
        """Validate if value is a positive integer"""
        try:
            num = int(value)
            return num > 0
        except ValueError:
            return False
    
    def validate_book_title(self, title):
        """Validate book title"""
        if len(title) < 1 or len(title) > 200:
            return False
        pattern = r'^[a-zA-Z0-9\s\-:,.\'\"!?&()]+$'
        return re.match(pattern, title) is not None
    
    def generate_qr_code(self, book_id, book_title):
        """Generate QR code for book"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr_data = f"LibraAI-Book-ID:{book_id}|Title:{book_title}"
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        qr_path = f"qr_codes/book_{book_id}.png"
        img.save(qr_path)
        return qr_path
    
    
    
    def hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_login(self):
        """Display modern login screen"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True)
        
        # Left side - Branding
        left_frame = tk.Frame(main_frame, bg="#16213e", width=600)
        left_frame.pack(side="left", fill="both", expand=True)
        
        branding_container = tk.Frame(left_frame, bg="#16213e")
        branding_container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(branding_container, text="📚", font=("Arial", 80), 
                bg="#16213e", fg="#0f3460").pack()
        tk.Label(branding_container, text="LibraAI", font=("Arial", 48, "bold"), 
                bg="#16213e", fg="#e94560").pack(pady=10)
        tk.Label(branding_container, text="AI-Powered Library Management", 
                font=("Arial", 16), bg="#16213e", fg="#ffffff").pack()
        tk.Label(branding_container, text="Intelligent • Efficient • Modern", 
                font=("Arial", 14, "italic"), bg="#16213e", fg="#a0a0a0").pack(pady=5)
        
        # Right side - Login form
        right_frame = tk.Frame(main_frame, bg="#0f3460", width=600)
        right_frame.pack(side="right", fill="both", expand=True)
        
        login_container = tk.Frame(right_frame, bg="#0f3460", padx=50, pady=40)
        login_container.place(relx=0.5, rely=0.5, anchor="center")
        
        tk.Label(login_container, text="Welcome Back!", font=("Arial", 32, "bold"),
                bg="#0f3460", fg="#ffffff").pack(pady=(0, 10))
        tk.Label(login_container, text="Please login to continue", 
                font=("Arial", 12), bg="#0f3460", fg="#a0a0a0").pack(pady=(0, 40))
        
        # Username
        tk.Label(login_container, text="Username", font=("Arial", 11, "bold"), 
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        username_entry = tk.Entry(login_container, font=("Arial", 12), width=30,
                                 bg="#1a1a2e", fg="#ffffff", insertbackground="#ffffff",
                                 relief="flat", bd=0)
        username_entry.pack(pady=(0, 20), ipady=8)
        
        # Password
        tk.Label(login_container, text="Password", font=("Arial", 11, "bold"), 
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        password_entry = tk.Entry(login_container, font=("Arial", 12), width=30, show="●",
                                 bg="#1a1a2e", fg="#ffffff", insertbackground="#ffffff",
                                 relief="flat", bd=0)
        password_entry.pack(pady=(0, 30), ipady=8)
        
        # Login button
        login_btn = tk.Button(login_container, text="LOGIN", font=("Arial", 12, "bold"),
                             bg="#e94560", fg="white", width=30, cursor="hand2",
                             relief="flat", bd=0, activebackground="#c72c41",
                             command=lambda: self.authenticate(username_entry.get(), 
                                                              password_entry.get()))
        login_btn.pack(pady=(0, 15), ipady=10)
        
        # Register link
        register_frame = tk.Frame(login_container, bg="#0f3460")
        register_frame.pack(pady=10)
        
        tk.Label(register_frame, text="Don't have an account? ", 
                font=("Arial", 10), bg="#0f3460", fg="#a0a0a0").pack(side="left")
        
        register_btn = tk.Button(register_frame, text="Register Here", 
                                font=("Arial", 10, "bold"), bg="#0f3460", fg="#e94560",
                                bd=0, cursor="hand2", activebackground="#0f3460",
                                command=self.show_register)
        register_btn.pack(side="left")
        
        password_entry.bind('<Return>', lambda e: self.authenticate(
            username_entry.get(), password_entry.get()))
        
        # Forgot password link
        forgot_frame = tk.Frame(login_container, bg="#0f3460")
        forgot_frame.pack(pady=5)
        
        forgot_btn = tk.Button(forgot_frame, text="Forgot Password?", 
                            font=("Arial", 10, "underline"), bg="#0f3460", fg="#3498db",
                            bd=0, cursor="hand2", activebackground="#0f3460",
                            command=self.show_forgot_password)
        forgot_btn.pack()
    
    def authenticate(self, username, password):
        """Authenticate user login"""
        if not username or not password:
            messagebox.showwarning("Input Error", "Please enter both username and password")
            return
        
        hashed_pw = self.hash_password(password)
        self.cursor.execute("""
            SELECT user_id, username, full_name, role 
            FROM users WHERE username = %s AND password = %s
        """, (username, hashed_pw))
        
        user = self.cursor.fetchone()
        
        if user:
            self.current_user = {
                'user_id': user[0],
                'username': user[1],
                'full_name': user[2],
                'role': user[3]
            }
            self.user_role = user[3]
            
            if self.user_role == 'librarian':
                self.show_librarian_dashboard()
            else:
                self.show_student_dashboard()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")
    
    def show_register(self):
        """Display enhanced registration form"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True)
        
        # Scrollable frame
        canvas = tk.Canvas(main_frame, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        register_frame = tk.Frame(scrollable_frame, bg="#0f3460", padx=60, pady=40)
        register_frame.pack(pady=50, padx=200)
        
        tk.Label(register_frame, text="🎓 Student Registration", 
                font=("Arial", 28, "bold"), bg="#0f3460", fg="#e94560").pack(pady=(0, 10))
        tk.Label(register_frame, text="Join LibraAI Community", 
                font=("Arial", 12), bg="#0f3460", fg="#a0a0a0").pack(pady=(0, 30))
        
        fields = [
            ("Full Name", "full_name", "Enter your full name", False),
            ("Username", "username", "Choose a username (4-20 characters)", False),
            ("Email Address", "email", "your.email@example.com", False),
            ("Phone Number", "phone", "01XXXXXXXXX", False),
            ("Password", "password", "Strong password required", True),
            ("Confirm Password", "confirm_password", "Re-enter password", True)
        ]
        
        entries = {}
        for label, key, placeholder, is_password in fields:
            field_frame = tk.Frame(register_frame, bg="#0f3460")
            field_frame.pack(fill="x", pady=10)
            
            tk.Label(field_frame, text=label, font=("Arial", 11, "bold"), 
                    bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 5))
            
            entry = tk.Entry(field_frame, font=("Arial", 11), width=45,
                           show="●" if is_password else "",
                           bg="#1a1a2e", fg="#ffffff", insertbackground="#ffffff",
                           relief="flat", bd=0)
            entry.pack(ipady=8)
            entry.insert(0, placeholder)
            entry.bind("<FocusIn>", lambda e, ent=entry, ph=placeholder: 
                      ent.delete(0, 'end') if ent.get() == ph else None)
            entries[key] = entry
        
        # Password strength indicator
        self.password_strength_label = tk.Label(register_frame, text="", 
                                               font=("Arial", 9), bg="#0f3460")
        self.password_strength_label.pack(pady=5)
        
        entries['password'].bind('<KeyRelease>', lambda e: self.check_password_strength(
            entries['password'].get(), self.password_strength_label))
        
        # Buttons
        btn_frame = tk.Frame(register_frame, bg="#0f3460")
        btn_frame.pack(pady=30)
        
        tk.Button(btn_frame, text="REGISTER", font=("Arial", 11, "bold"),
                 bg="#27ae60", fg="white", width=20, cursor="hand2",
                 relief="flat", activebackground="#229954",
                 command=lambda: self.register_user(entries)).pack(side="left", padx=5, ipady=10)
        
        tk.Button(btn_frame, text="BACK TO LOGIN", font=("Arial", 11, "bold"),
                 bg="#95a5a6", fg="white", width=20, cursor="hand2",
                 relief="flat", activebackground="#7f8c8d",
                 command=self.show_login).pack(side="left", padx=5, ipady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def check_password_strength(self, password, label):
        """Display password strength in real-time"""
        if not password or password.startswith("Strong password"):
            label.config(text="", fg="")
            return
        
        is_valid, message = self.validate_password(password)
        if is_valid:
            label.config(text="✓ " + message, fg="#27ae60")
        else:
            label.config(text="✗ " + message, fg="#e74c3c")
    
    def register_user(self, entries):
        """Register new student with comprehensive validation"""
        data = {}
        for key, entry in entries.items():
            value = entry.get().strip()
            if value in ["Enter your full name", "Choose a username (4-20 characters)", 
                        "your.email@example.com", "01XXXXXXXXX", 
                        "Strong password required", "Re-enter password", ""]:
                messagebox.showwarning("Input Error", f"Please fill in {key.replace('_', ' ')}")
                return
            data[key] = value
        
        # Validate full name
        if not self.validate_name(data['full_name']):
            messagebox.showerror("Validation Error", "Full name should contain only letters and spaces")
            return
        
        # Validate username
        if not self.validate_username(data['username']):
            messagebox.showerror("Validation Error", 
                            "Username must be 4-20 characters and contain only letters, numbers, and underscores")
            return
        
        # Validate email
        if not self.validate_email(data['email']):
            messagebox.showerror("Validation Error", "Please enter a valid email address")
            return
        
        # Validate phone
        if not self.validate_phone(data['phone']):
            messagebox.showerror("Validation Error", 
                            "Please enter a valid Malaysian phone number (e.g., 0123456789)")
            return
        
        # Validate password
        is_valid, message = self.validate_password(data['password'])
        if not is_valid:
            messagebox.showerror("Validation Error", message)
            return
        
        # Check password match
        if data['password'] != data['confirm_password']:
            messagebox.showerror("Error", "Passwords do not match")
            return
        
        # Check if username exists
        self.cursor.execute("SELECT * FROM users WHERE username = %s", (data['username'],))
        if self.cursor.fetchone():
            messagebox.showerror("Error", "Username already exists")
            return
        
        # Check if email exists
        self.cursor.execute("SELECT * FROM users WHERE email = %s", (data['email'],))
        if self.cursor.fetchone():
            messagebox.showerror("Error", "Email already registered")
            return
        
        # Insert user (ONLY ONCE!)
        try:
            hashed_pw = self.hash_password(data['password'])
            self.cursor.execute("""
                INSERT INTO users (username, password, full_name, email, phone, role)
                VALUES (%s, %s, %s, %s, %s, 'student')
            """, (data['username'], hashed_pw, data['full_name'], data['email'], data['phone']))
            self.db.commit()
            
            # SEND WELCOME EMAIL
            try:
                self.send_welcome_email(data['email'], data['full_name'], data['username'])
            except Exception as email_error:
                print(f"Warning: Failed to send welcome email: {email_error}")
            
            messagebox.showinfo("Success", 
                "Registration successful!\n\nA welcome email has been sent.\n\nPlease login with your credentials.")
            self.show_login()
            
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error: {err}")
    def show_student_dashboard(self):
        """Display modern student dashboard"""
        self.clear_window()
        
        # Generate notifications
        self.generate_notifications()
        
        # Header
        header = tk.Frame(self.root, bg="#e94560", height=80)
        header.pack(fill="x")
        
        header_left = tk.Frame(header, bg="#e94560")
        header_left.pack(side="left", padx=20, pady=15)
        
        tk.Label(header_left, text="📚 LibraAI", font=("Arial", 24, "bold"), 
                bg="#e94560", fg="#ffffff").pack(anchor="w")
        tk.Label(header_left, text="Student Portal", font=("Arial", 12), 
                bg="#e94560", fg="#ffffff").pack(anchor="w")
        
        # ADD QUICK ACTION BUTTONS (NEW):
        header_right = tk.Frame(header, bg="#e94560")
        header_right.pack(side="right", padx=20, pady=15)
        
        tk.Button(header_right, text="📱 Quick Borrow (QR)", font=("Arial", 10, "bold"),
                bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_quick_borrow_by_id).pack(side="left", padx=5, ipady=8, ipadx=12)
        
        tk.Button(header_right, text="📱 Quick Return (QR)", font=("Arial", 10, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_quick_return_by_id).pack(side="left", padx=5, ipady=8, ipadx=12)
        
        tk.Label(header, text=f"Welcome, {self.current_user['full_name']} 👋", 
                font=("Arial", 14, "bold"), bg="#e94560", fg="#ffffff").pack(side="right", padx=20)
        
        # ... rest of existing code (notification banner, sidebar, etc.) ...
        

        
        # ================================================================
        # NOTIFICATION BANNER (NEW)
        # ================================================================
        # Check for unread notifications
        self.cursor.execute("""
            SELECT COUNT(*) FROM notifications 
            WHERE user_id = %s AND is_read = 0
        """, (self.current_user['user_id'],))
        unread_count = self.cursor.fetchone()[0]
        
        if unread_count > 0:
            # Get most urgent notification
            self.cursor.execute("""
                SELECT title, message, type FROM notifications
                WHERE user_id = %s AND is_read = 0
                ORDER BY FIELD(type, 'overdue', 'due_soon', 'penalty', 'info'), created_at DESC
                LIMIT 1
            """, (self.current_user['user_id'],))
            
            notif = self.cursor.fetchone()
            if notif:
                title, message, ntype = notif
                
                # Color based on type
                colors = {
                    'overdue': '#e74c3c',
                    'due_soon': '#f39c12',
                    'penalty': '#e67e22',
                    'info': '#3498db'
                }
                bg_color = colors.get(ntype, '#3498db')
                
                banner = tk.Frame(self.root, bg=bg_color, height=60)
                banner.pack(fill="x")
                
                banner_content = tk.Frame(banner, bg=bg_color)
                banner_content.pack(fill="both", expand=True, padx=20, pady=10)
                
                icon = "⚠️" if ntype == 'overdue' else "🔔"
                tk.Label(banner_content, text=icon, font=("Arial", 20),
                        bg=bg_color, fg="white").pack(side="left", padx=10)
                
                text_frame = tk.Frame(banner_content, bg=bg_color)
                text_frame.pack(side="left", fill="x", expand=True)
                
                tk.Label(text_frame, text=title, font=("Arial", 12, "bold"),
                        bg=bg_color, fg="white").pack(anchor="w")
                tk.Label(text_frame, text=f"{unread_count} unread notification(s)", 
                        font=("Arial", 9),
                        bg=bg_color, fg="white").pack(anchor="w")
                
                tk.Button(banner_content, text="VIEW ALL", font=("Arial", 10, "bold"),
                        bg="white", fg=bg_color, cursor="hand2", relief="flat", bd=0,
                        command=self.show_notifications).pack(side="right", padx=10, ipady=5, ipadx=15)
        # ================================================================
        
        # Sidebar
        sidebar = tk.Frame(self.root, bg="#16213e", width=250)
        sidebar.pack(side="left", fill="y")
        
        menu_items = [
            ("🔍 AI Search Books", self.show_search_books, "#1a1a2e"),
            ("🤖 AI Recommendations", self.show_recommendations, "#1a1a2e"),
            ("📂 Browse by Category", self.show_browse_by_category, "#1a1a2e"),
            ("📖 My Borrowed Books", self.show_my_books, "#1a1a2e"),
            ("📋 My Reservations", self.show_my_reservations, "#1a1a2e"),
            ("💰 My Penalties", self.show_my_penalties, "#1a1a2e"),
            ("🔔 Notifications", self.show_notifications, "#1a1a2e"),  # NEW
            ("👤 My Profile", self.show_student_profile, "#1a1a2e"),
            ("🚪 Logout", self.show_login, "#c0392b")
        ]
        
        tk.Label(sidebar, text="MENU", font=("Arial", 10, "bold"), 
                bg="#16213e", fg="#7f8c8d").pack(pady=20, padx=20, anchor="w")
        
        for text, command, color in menu_items:
            # Add badge for notifications
            if "Notifications" in text and unread_count > 0:
                btn_frame = tk.Frame(sidebar, bg="#16213e")
                btn_frame.pack(fill="x", pady=3, padx=10)
                
                btn = tk.Button(btn_frame, text=text, font=("Arial", 12, "bold"), 
                            bg=color, fg="white", bd=0, cursor="hand2", 
                            anchor="w", command=command,
                            activebackground="#0f3460")
                btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
                
                badge = tk.Label(btn_frame, text=str(unread_count), font=("Arial", 9, "bold"),
                            bg="#e74c3c", fg="white", width=3)
                badge.pack(side="right", padx=5)
            else:
                btn = tk.Button(sidebar, text=text, font=("Arial", 12, "bold"), 
                            bg=color, fg="white", bd=0, cursor="hand2", 
                            anchor="w", padx=25, command=command,
                            activebackground="#0f3460")
                btn.pack(fill="x", pady=3, padx=10)
        
        # Main content
        self.content_frame = tk.Frame(self.root, bg="#0f3460")
        self.content_frame.pack(side="right", fill="both", expand=True)
        
        self.show_search_books()
        
        # Show popup if critical notifications
        self.show_notification_popup()
    
    
    def show_librarian_dashboard(self):
        """Display modern librarian dashboard"""
        self.clear_window()
        
        # Header
        header = tk.Frame(self.root, bg="#8e44ad", height=80)
        header.pack(fill="x")
        
        header_left = tk.Frame(header, bg="#8e44ad")
        header_left.pack(side="left", padx=20, pady=15)
        
        tk.Label(header_left, text="📚 LibraAI", font=("Arial", 24, "bold"), 
                bg="#8e44ad", fg="#ffffff").pack(anchor="w")
        tk.Label(header_left, text="Librarian Portal", font=("Arial", 12), 
                bg="#8e44ad", fg="#ffffff").pack(anchor="w")
        
        tk.Label(header, text=f"Welcome, {self.current_user['full_name']} 👋", 
                font=("Arial", 14, "bold"), bg="#8e44ad", fg="#ffffff").pack(side="right", padx=20)
        
        # Sidebar
        sidebar = tk.Frame(self.root, bg="#16213e", width=250)
        sidebar.pack(side="left", fill="y")
        
        menu_items = [
            ("📚 Manage Books", self.show_manage_books, "#1a1a2e"),
            ("📑 Manage Categories", self.show_manage_categories, "#1a1a2e"),
            ("👥 Manage Users", self.show_manage_users, "#1a1a2e"),
            ("📋 Manage Reservations", self.show_manage_reservations, "#1a1a2e"),
            ("💰 Manage Penalties", self.show_manage_penalties, "#1a1a2e"),
            ("📊 View Transactions", self.show_transactions, "#1a1a2e"),
            ("📈 Generate Reports", self.show_reports, "#1a1a2e"),
            ("🚪 Logout", self.show_login, "#c0392b")
        ]
        
        tk.Label(sidebar, text="ADMIN MENU", font=("Arial", 10, "bold"), 
                bg="#16213e", fg="#7f8c8d").pack(pady=20, padx=20, anchor="w")
        
        for text, command, color in menu_items:
            btn = tk.Button(sidebar, text=text, font=("Arial", 12, "bold"), 
                          bg=color, fg="white", bd=0, cursor="hand2", 
                          anchor="w", padx=25, command=command,
                          activebackground="#0f3460")
            btn.pack(fill="x", pady=3, padx=10)
        
        # Main content
        self.content_frame = tk.Frame(self.root, bg="#0f3460")
        self.content_frame.pack(side="right", fill="both", expand=True)
        
        self.show_manage_books()
    
    def show_search_books(self):
        """Display AI-powered book search interface with category filter"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="🤖 AI-Powered Book Search", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Intelligent search with fuzzy matching and relevance scoring", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Search frame with category filter
        search_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        search_frame.pack(fill="x", padx=30, pady=10)
        
        search_container = tk.Frame(search_frame, bg="#1a1a2e")
        search_container.pack(pady=15)
        
        tk.Label(search_container, text="🔍", font=("Arial", 20), 
                bg="#1a1a2e", fg="#e94560").pack(side="left", padx=10)
        
        # Search text entry
        search_entry = tk.Entry(search_container, font=("Arial", 13), width=40,
                            bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        search_entry.pack(side="left", ipady=10, padx=5)
        
        # Category filter dropdown
        tk.Label(search_container, text="📂", font=("Arial", 16), 
                bg="#1a1a2e", fg="#e94560").pack(side="left", padx=(10, 5))
        
        # Get categories from database
        self.cursor.execute("SELECT category_name FROM categories ORDER BY category_name ASC")
        categories = ["All Categories"] + [row[0] for row in self.cursor.fetchall()]
        
        category_filter = ttk.Combobox(search_container, font=("Arial", 11), width=20,
                                    values=categories, state="readonly")
        category_filter.set("All Categories")
        category_filter.pack(side="left", padx=5, ipady=10)
        
        # Search button
        tk.Button(search_container, text="SEARCH", font=("Arial", 11, "bold"),
                bg="#e94560", fg="white", cursor="hand2", relief="flat", bd=0,
                command=lambda: self.display_search_results_with_filter(
                    search_entry.get(), category_filter.get())).pack(
                        side="left", padx=5, ipady=10, ipadx=20)
        
        # Show all button
        tk.Button(search_container, text="SHOW ALL", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=lambda: self.display_search_results_with_filter("", "All Categories")).pack(
                    side="left", padx=5, ipady=10, ipadx=20)
        
        # Results container
        self.results_container = tk.Frame(self.content_frame, bg="#0f3460")
        self.results_container.pack(fill="both", expand=True, padx=30, pady=10)
        
        # Initial load - show all books
        self.display_search_results_with_filter("", "All Categories")
    
    # 17. UPDATE SEARCH FUNCTION TO INCLUDE CATEGORY FILTER
    def display_search_results_with_filter(self, query, category):
        """Display search results with category filter"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(self.results_container, bg="#0f3460", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.results_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f3460")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Get search results with category filter
        results = self.ai_search_books_with_category(query, category)
        
        if not results:
            tk.Label(scrollable_frame, text="😔 No books found matching your search", 
                    font=("Arial", 16), bg="#0f3460", fg="#a0a0a0").pack(pady=50)
        else:
            # Show result count with filter info
            filter_text = f" in category '{category}'" if category != "All Categories" else ""
            tk.Label(scrollable_frame, text=f"Found {len(results)} book(s){filter_text}", 
                    font=("Arial", 12, "bold"), bg="#0f3460", fg="#27ae60").pack(
                        anchor="w", padx=10, pady=10)
            
            # Display books in grid (3 per row)
            row_frame = None
            for idx, book in enumerate(results):
                if idx % 3 == 0:
                    row_frame = tk.Frame(scrollable_frame, bg="#0f3460")
                    row_frame.pack(fill="x", padx=10, pady=10)
                
                self.create_book_card(row_frame, book)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    
    def create_book_card(self, parent, book):
        """Create a modern book card"""
        book_id, title, author, category, available, location, image_path = book[:7]
        
        card = tk.Frame(parent, bg="#1a1a2e", width=280, height=500, relief="flat", bd=0)
        card.pack(side="left", padx=10, pady=10)
        card.pack_propagate(False)
        
        # Book image
        img_frame = tk.Frame(card, bg="#0f3460", width=260, height=200)
        img_frame.pack(pady=10, padx=10)
        img_frame.pack_propagate(False)
        
        try:
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((260, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = tk.Label(img_frame, image=photo, bg="#0f3460")
                img_label.image = photo
                img_label.pack()
            else:
                tk.Label(img_frame, text="📚", font=("Arial", 60), 
                        bg="#0f3460", fg="#e94560").pack(expand=True)
        except:
            tk.Label(img_frame, text="📚", font=("Arial", 60), 
                    bg="#0f3460", fg="#e94560").pack(expand=True)
        
        # Book info
        info_frame = tk.Frame(card, bg="#1a1a2e")
        info_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Title
        title_text = title if len(title) <= 30 else title[:27] + "..."
        tk.Label(info_frame, text=title_text, font=("Arial", 12, "bold"), 
                bg="#1a1a2e", fg="#ffffff", wraplength=250).pack(anchor="w")
        
        # Author
        tk.Label(info_frame, text=f"by {author}", font=("Arial", 10, "italic"), 
                bg="#1a1a2e", fg="#a0a0a0").pack(anchor="w", pady=2)
        
        # Category
        tk.Label(info_frame, text=f"📖 {category}", font=("Arial", 9), 
                bg="#1a1a2e", fg="#7f8c8d").pack(anchor="w", pady=2)
        
        # Availability
        avail_text = f"✓ {available} Available" if available > 0 else "✗ Not Available"
        avail_color = "#27ae60" if available > 0 else "#e74c3c"
        tk.Label(info_frame, text=avail_text, font=("Arial", 10, "bold"), 
                bg="#1a1a2e", fg=avail_color).pack(anchor="w", pady=5)
        
        # Borrow/Reserve button
        if available > 0:
            tk.Button(info_frame, text="BORROW", font=("Arial", 10, "bold"),
                    bg="#e94560", fg="white", cursor="hand2", relief="flat", bd=0,
                    command=lambda: self.borrow_book_by_id(book_id)).pack(fill="x", pady=5, ipady=8)
        else:
            # Check if already reserved by this user
            self.cursor.execute("""
                SELECT reservation_id FROM reservations 
                WHERE user_id = %s AND book_id = %s AND status = 'pending'
            """, (self.current_user['user_id'], book_id))
            
            if self.cursor.fetchone():
                tk.Button(info_frame, text="RESERVED ✓", font=("Arial", 10, "bold"),
                        bg="#f39c12", fg="white", state="disabled", relief="flat", bd=0).pack(
                            fill="x", pady=5, ipady=8)
            else:
                tk.Button(info_frame, text="RESERVE", font=("Arial", 10, "bold"),
                        bg="#9b59b6", fg="white", cursor="hand2", relief="flat", bd=0,
                        command=lambda: self.reserve_book(book_id)).pack(fill="x", pady=5, ipady=8)
        
        # QR Code button - ALWAYS show for all books (NOT indented under if/else!)
        tk.Button(info_frame, text="📱 VIEW QR CODE", font=("Arial", 9, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=lambda: self.show_book_qr_code_student(book_id)).pack(fill="x", pady=5, ipady=6)

    
    def borrow_book_by_id(self, book_id):
        """Borrow book by ID"""
        # Check availability
        self.cursor.execute("SELECT title, available FROM books WHERE book_id = %s", (book_id,))
        result = self.cursor.fetchone()
        
        if not result:
            messagebox.showerror("Error", "Book not found")
            return
        
        title, available = result
        
        if available <= 0:
            messagebox.showerror("Error", "This book is not available")
            return
        
        # Check if user already borrowed this book
        self.cursor.execute("""
            SELECT * FROM transactions 
            WHERE user_id = %s AND book_id = %s AND status = 'borrowed'
        """, (self.current_user['user_id'], book_id))
        
        if self.cursor.fetchone():
            messagebox.showerror("Error", "You have already borrowed this book")
            return
        
        # Create transaction
        borrow_date = datetime.now().date()
        due_date = borrow_date + timedelta(days=14)
        
        self.cursor.execute("""
            INSERT INTO transactions (user_id, book_id, borrow_date, due_date, status)
            VALUES (%s, %s, %s, %s, 'borrowed')
        """, (self.current_user['user_id'], book_id, borrow_date, due_date))
        
        # Update book availability
        self.cursor.execute("""
            UPDATE books SET available = available - 1 WHERE book_id = %s
        """, (book_id,))
        
        self.db.commit()
        messagebox.showinfo("Success", f"'{title}' borrowed successfully!\n\nDue date: {due_date}\nPlease return on time to avoid penalties.")
        self.show_search_books()
    
    def show_my_books(self):
        """Display borrowed books with modern UI"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📖 My Borrowed Books", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Manage your borrowed books and track due dates", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Table frame
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        # Treeview with style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.Treeview", 
                       background="#1a1a2e",
                       foreground="white",
                       fieldbackground="#1a1a2e",
                       borderwidth=0)
        style.configure("Custom.Treeview.Heading",
                       background="#e94560",
                       foreground="white",
                       borderwidth=0)
        style.map("Custom.Treeview",
                 background=[("selected", "#e94560")])
        
        columns = ('Trans ID', 'Book Title', 'Borrow Date', 'Due Date', 'Status')
        self.borrowed_tree = ttk.Treeview(table_frame, columns=columns, show='headings', 
                                         height=15, style="Custom.Treeview")
        
        for col in columns:
            self.borrowed_tree.heading(col, text=col)
            self.borrowed_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.borrowed_tree.yview)
        self.borrowed_tree.configure(yscrollcommand=scrollbar.set)
        
        self.borrowed_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Load borrowed books
        self.cursor.execute("""
            SELECT t.transaction_id, b.title, t.borrow_date, t.due_date, t.status
            FROM transactions t
            JOIN books b ON t.book_id = b.book_id
            WHERE t.user_id = %s
            ORDER BY t.borrow_date DESC
        """, (self.current_user['user_id'],))
        
        for row in self.cursor.fetchall():
            self.borrowed_tree.insert('', 'end', values=row)
        
        # Button frame
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="📥 RETURN BOOK", font=("Arial", 11, "bold"),
                 bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.return_book).pack(side="left", padx=10, ipady=12, ipadx=30)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                 bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.show_my_books).pack(side="left", padx=10, ipady=12, ipadx=30)
        
        tk.Button(btn_frame, text="📱 VIEW QR CODE", font=("Arial", 11, "bold"),
                 bg="#9b59b6", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.view_borrowed_book_qr).pack(side="left", padx=10, ipady=12, ipadx=30)
    
    def return_book(self):
        """Return borrowed book"""
        selected = self.borrowed_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a book to return")
            return
        
        trans_id = self.borrowed_tree.item(selected[0])['values'][0]
        
        # Get transaction details
        self.cursor.execute("""
            SELECT book_id, due_date, status FROM transactions WHERE transaction_id = %s
        """, (trans_id,))
        
        result = self.cursor.fetchone()
        if not result:
            return
        
        book_id, due_date, status = result
        
        if status == 'returned':
            messagebox.showinfo("Info", "This book has already been returned")
            return
        
        return_date = datetime.now().date()
        


        # Calculate penalty if overdue
        penalty_message = ""
        if return_date > due_date:
            days_overdue = (return_date - due_date).days
            penalty_amount = days_overdue * 1.00
            
            self.cursor.execute("""
                INSERT INTO penalties (transaction_id, user_id, amount, days_overdue)
                VALUES (%s, %s, %s, %s)
            """, (trans_id, self.current_user['user_id'], penalty_amount, days_overdue))
            
            penalty_message = f"\n\n⚠️ OVERDUE NOTICE\nDays overdue: {days_overdue}\nPenalty: RM {penalty_amount:.2f}\n\nPlease settle the penalty at the library counter."
        

        if return_date > due_date:
            days_overdue = (return_date - due_date).days
            penalty_amount = days_overdue * 1.00
            
            self.cursor.execute("""
                INSERT INTO penalties (transaction_id, user_id, amount, days_overdue)
                VALUES (%s, %s, %s, %s)
            """, (trans_id, self.current_user['user_id'], penalty_amount, days_overdue))
            
            # SEND PENALTY EMAIL (NEW)
            self.cursor.execute("""
                SELECT b.title, u.email, u.full_name
                FROM transactions t
                JOIN books b ON t.book_id = b.book_id
                JOIN users u ON t.user_id = u.user_id
                WHERE t.transaction_id = %s
            """, (trans_id,))
            
            book_title, user_email, user_name = self.cursor.fetchone()
            if user_email:
                self.send_penalty_notice_email(user_email, user_name, book_title, penalty_amount)

        # Update transaction
        self.cursor.execute("""
            UPDATE transactions 
            SET return_date = %s, status = 'returned'
            WHERE transaction_id = %s
        """, (return_date, trans_id))
        
        # Update book availability
        self.cursor.execute("""
            UPDATE books SET available = available + 1 WHERE book_id = %s
        """, (book_id,))
        
        self.db.commit()

        self.check_available_reservations()
        message = "✅ Book returned successfully!" + penalty_message
        if penalty_message:
            messagebox.showwarning("Book Returned", message)
        else:
            messagebox.showinfo("Success", message)
        
        self.show_my_books()
    
    def show_my_penalties(self):
        """Display penalties with modern UI"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="💰 My Penalties", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="View and track your penalty records", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Summary cards
        summary_frame = tk.Frame(self.content_frame, bg="#0f3460")
        summary_frame.pack(fill="x", padx=30, pady=10)
        
        # Get statistics
        self.cursor.execute("""
            SELECT SUM(amount) FROM penalties WHERE user_id = %s AND paid = 0
        """, (self.current_user['user_id'],))
        total_unpaid = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("""
            SELECT SUM(amount) FROM penalties WHERE user_id = %s AND paid = 1
        """, (self.current_user['user_id'],))
        total_paid = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM penalties WHERE user_id = %s
        """, (self.current_user['user_id'],))
        total_penalties = self.cursor.fetchone()[0]
        
        stats = [
            ("💳 Total Outstanding", f"RM {total_unpaid:.2f}", "#e74c3c"),
            ("✅ Total Paid", f"RM {total_paid:.2f}", "#27ae60"),
            ("📊 Total Penalties", str(total_penalties), "#3498db")
        ]
        
        for label, value, color in stats:
            card = tk.Frame(summary_frame, bg=color, width=250, height=100)
            card.pack(side="left", padx=10, pady=10)
            card.pack_propagate(False)
            
            tk.Label(card, text=label, font=("Arial", 12, "bold"), 
                    bg=color, fg="white").pack(pady=(20, 5))
            tk.Label(card, text=value, font=("Arial", 20, "bold"), 
                    bg=color, fg="white").pack(pady=(0, 20))
        
        # Table frame
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        columns = ('Penalty ID', 'Days Overdue', 'Amount (RM)', 'Payment Date', 'Status')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', 
                        height=12, style="Custom.Treeview")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Load penalties
        self.cursor.execute("""
            SELECT penalty_id, days_overdue, amount, payment_date,
                CASE WHEN paid = 1 THEN 'Paid ✓' ELSE 'Unpaid ✗' END as status
            FROM penalties
            WHERE user_id = %s
            ORDER BY penalty_id DESC
        """, (self.current_user['user_id'],))
        
        for row in self.cursor.fetchall():
            penalty_id, days, amount, pay_date, status = row
            display_date = pay_date if pay_date else "-"
            tree.insert('', 'end', values=(penalty_id, days, f"{amount:.2f}", display_date, status))
    
    def show_manage_books(self):
        """Display book management with enhanced UI"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📚 Manage Books", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Add, edit, and manage library book collection", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Action buttons
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Button(btn_frame, text="➕ ADD NEW BOOK", font=("Arial", 11, "bold"),
                 bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.show_add_book_form).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="✏️ EDIT BOOK", font=("Arial", 11, "bold"),
                 bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.edit_book).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🗑️ DELETE BOOK", font=("Arial", 11, "bold"),
                 bg="#e74c3c", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.delete_book).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="📱 VIEW QR CODE", font=("Arial", 11, "bold"),
                 bg="#9b59b6", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.show_qr_code).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                 bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                 command=self.load_all_books).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        # Table frame
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        columns = ('ID', 'ISBN', 'Title', 'Author', 'Category', 'Qty', 'Avail', 'Location')
        self.manage_books_tree = ttk.Treeview(table_frame, columns=columns, 
                                             show='headings', height=15, style="Custom.Treeview")
        
        for col in columns:
            self.manage_books_tree.heading(col, text=col)
            width = 250 if col == 'Title' else 100
            self.manage_books_tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", 
                                 command=self.manage_books_tree.yview)
        self.manage_books_tree.configure(yscrollcommand=scrollbar.set)
        
        self.manage_books_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        self.load_all_books()
    
    def load_all_books(self):
        """Load all books for librarian view"""
        for item in self.manage_books_tree.get_children():
            self.manage_books_tree.delete(item)
        
        self.cursor.execute("""
            SELECT book_id, isbn, title, author, category, quantity, available, location
            FROM books ORDER BY book_id DESC
        """)
        
        for row in self.cursor.fetchall():
            self.manage_books_tree.insert('', 'end', values=row)
    
    def show_add_book_form(self):
        """Show enhanced add book form with image upload"""
        form_window = tk.Toplevel(self.root)
        form_window.title("Add New Book")
        form_window.geometry("600x750")
        form_window.configure(bg="#1a1a2e")
        form_window.grab_set()
        
        # Scrollable frame
        canvas = tk.Canvas(form_window, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Header
        header = tk.Frame(scrollable_frame, bg="#8e44ad")
        header.pack(fill="x")
        
        tk.Label(header, text="➕ Add New Book", font=("Arial", 20, "bold"),
                bg="#8e44ad", fg="white").pack(pady=20)
        
        # Form container
        form_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Image upload section
        img_frame = tk.Frame(form_frame, bg="#0f3460", width=200, height=250)
        img_frame.pack(pady=20)
        img_frame.pack_propagate(False)
        
        self.selected_image_path = None
        self.img_preview_label = tk.Label(img_frame, text="📷\nClick to Upload\nBook Cover", 
                                        font=("Arial", 14), bg="#0f3460", fg="#a0a0a0",
                                        cursor="hand2")
        self.img_preview_label.pack(expand=True)
        self.img_preview_label.bind("<Button-1>", lambda e: self.select_book_image())
        
        tk.Label(form_frame, text="Supported: JPG, PNG (Max 5MB)", 
                font=("Arial", 9), bg="#1a1a2e", fg="#7f8c8d").pack()
        
        # ==============================================================
        # MODIFIED SECTION: Form fields with Category Dropdown
        # ==============================================================
        
        entries = {}
        
        # ISBN Field
        field_container = tk.Frame(form_frame, bg="#1a1a2e")
        field_container.pack(fill="x", pady=10)
        
        tk.Label(field_container, text="ISBN", 
                font=("Arial", 11, "bold"), bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        isbn_entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                    bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                    relief="flat", bd=0)
        isbn_entry.pack(ipady=10)
        isbn_entry.insert(0, "13-digit ISBN number")
        isbn_entry.bind("<FocusIn>", lambda e: isbn_entry.delete(0, 'end') 
                    if isbn_entry.get() == "13-digit ISBN number" else None)
        entries['isbn'] = isbn_entry
        
        # Title Field (Required)
        field_container = tk.Frame(form_frame, bg="#1a1a2e")
        field_container.pack(fill="x", pady=10)
        
        tk.Label(field_container, text="Title *", 
                font=("Arial", 11, "bold"), bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        title_entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                    bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                    relief="flat", bd=0)
        title_entry.pack(ipady=10)
        title_entry.insert(0, "Book title")
        title_entry.bind("<FocusIn>", lambda e: title_entry.delete(0, 'end') 
                        if title_entry.get() == "Book title" else None)
        entries['title'] = title_entry
        
        # Author Field (Required)
        field_container = tk.Frame(form_frame, bg="#1a1a2e")
        field_container.pack(fill="x", pady=10)
        
        tk.Label(field_container, text="Author *", 
                font=("Arial", 11, "bold"), bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        author_entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                    bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                    relief="flat", bd=0)
        author_entry.pack(ipady=10)
        author_entry.insert(0, "Author name")
        author_entry.bind("<FocusIn>", lambda e: author_entry.delete(0, 'end') 
                        if author_entry.get() == "Author name" else None)
        entries['author'] = author_entry
        
        # ============================================================
        # CATEGORY DROPDOWN (REPLACED TEXT ENTRY)
        # ============================================================
        field_container = tk.Frame(form_frame, bg="#1a1a2e")
        field_container.pack(fill="x", pady=10)
        
        tk.Label(field_container, text="Category", 
                font=("Arial", 11, "bold"), bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        # Get categories from database
        self.cursor.execute("SELECT category_name FROM categories ORDER BY category_name ASC")
        categories = [row[0] for row in self.cursor.fetchall()]
        
        # If no categories exist, add a default
        if not categories:
            categories = ["Fiction", "Science", "History", "Technology", "Other"]
        
        category_combo = ttk.Combobox(field_container, font=("Arial", 11), width=48,
                                    values=categories, state="readonly")
        category_combo.pack(ipady=10)
        if categories:
            category_combo.set(categories[0])  # Set first category as default
        
        entries['category'] = category_combo
        # ============================================================
        
        # Quantity Field (Required)
        field_container = tk.Frame(form_frame, bg="#1a1a2e")
        field_container.pack(fill="x", pady=10)
        
        tk.Label(field_container, text="Quantity *", 
                font=("Arial", 11, "bold"), bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        quantity_entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                    bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                    relief="flat", bd=0)
        quantity_entry.pack(ipady=10)
        quantity_entry.insert(0, "Number of copies")
        quantity_entry.bind("<FocusIn>", lambda e: quantity_entry.delete(0, 'end') 
                        if quantity_entry.get() == "Number of copies" else None)
        entries['quantity'] = quantity_entry
        
        # Location Field
        field_container = tk.Frame(form_frame, bg="#1a1a2e")
        field_container.pack(fill="x", pady=10)
        
        tk.Label(field_container, text="Location", 
                font=("Arial", 11, "bold"), bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        location_entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                    bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                    relief="flat", bd=0)
        location_entry.pack(ipady=10)
        location_entry.insert(0, "Shelf location (e.g., A-12)")
        location_entry.bind("<FocusIn>", lambda e: location_entry.delete(0, 'end') 
                        if location_entry.get() == "Shelf location (e.g., A-12)" else None)
        entries['location'] = location_entry
        
        # ==============================================================
        # END OF MODIFIED SECTION
        # ==============================================================
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ ADD BOOK", font=("Arial", 12, "bold"),
                bg="#27ae60", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0, activebackground="#229954",
                command=lambda: self.add_book(entries, form_window)).pack(side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=form_window.destroy).pack(side="left", padx=10, ipady=12)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def select_book_image(self):
        """Select and validate book cover image"""
        file_path = filedialog.askopenfilename(
            title="Select Book Cover Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if file_path:
            # Validate file type
            if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                messagebox.showerror("Invalid File", "Please select a valid image file (JPG or PNG)")
                return
            
            # Validate file size (5MB max)
            if os.path.getsize(file_path) > 5 * 1024 * 1024:
                messagebox.showerror("File Too Large", "Image size must be less than 5MB")
                return
            
            self.selected_image_path = file_path
            
            # Show preview
            try:
                img = Image.open(file_path)
                img = img.resize((200, 250), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.img_preview_label.config(image=photo, text="")
                self.img_preview_label.image = photo
            except:
                messagebox.showerror("Error", "Failed to load image")
    
    def add_book(self, entries, window):
        """Add new book with comprehensive validation"""
        data = {}
        # Get values from entries
        for key, entry in entries.items():
            if key == 'category':
                # Category is a Combobox, just get its value
                value = entry.get().strip()
            else:
                # Regular Entry fields
                value = entry.get().strip()
                placeholders = ["13-digit ISBN number", "Book title", "Author name", 
                            "Number of copies", "Shelf location (e.g., A-12)", ""]
                if value in placeholders:
                    if key in ['title', 'author', 'quantity']:
                        messagebox.showwarning("Input Error", f"Please fill in {key.replace('_', ' ')}")
                        return
                    value = ""
            data[key] = value
        
        # Validate required fields
        if not data['title'] or not data['author'] or not data['quantity']:
            messagebox.showwarning("Input Error", "Please fill in all required fields (Title, Author, Quantity)")
            return
        
        # Validate title
        if not self.validate_book_title(data['title']):
            messagebox.showerror("Validation Error", "Title contains invalid characters or is too long")
            return
        
        # Validate author name
        if not self.validate_name(data['author']):
            messagebox.showerror("Validation Error", "Author name should contain only letters and spaces")
            return
        
        # Validate ISBN if provided
        if data['isbn'] and not self.validate_isbn(data['isbn']):
            messagebox.showerror("Validation Error", "Invalid ISBN format. Must be 10 or 13 digits")
            return
        
        # Validate quantity
        if not self.validate_number(data['quantity']):
            messagebox.showerror("Validation Error", "Quantity must be a positive integer")
            return
        
        # Check if ISBN exists
        if data['isbn']:
            self.cursor.execute("SELECT * FROM books WHERE isbn = %s", (data['isbn'],))
            if self.cursor.fetchone():
                messagebox.showerror("Error", "A book with this ISBN already exists")
                return
        
        # Insert book
        try:
            quantity = int(data['quantity'])
            
            # Copy image if selected
            image_path = None
            if self.selected_image_path:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                ext = os.path.splitext(self.selected_image_path)[1]
                new_filename = f"book_{timestamp}{ext}"
                dest_path = os.path.join("book_images", new_filename)
                shutil.copy2(self.selected_image_path, dest_path)
                image_path = dest_path
            
            # Category from dropdown (no need to check if empty, dropdown always has a value)
            category_value = data['category'] if data['category'] else None
            
            self.cursor.execute("""
                INSERT INTO books (isbn, title, author, category, quantity, available, location, image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (data['isbn'] if data['isbn'] else None, 
                data['title'], 
                data['author'], 
                category_value,
                quantity, 
                quantity, 
                data['location'] if data['location'] else None,
                image_path))
            
            self.db.commit()
            
            # Get the inserted book ID
            book_id = self.cursor.lastrowid
            
            # Generate QR code
            qr_path = self.generate_qr_code(book_id, data['title'])
            
            # Update book with QR code path
            self.cursor.execute("""
                UPDATE books SET qr_code_path = %s WHERE book_id = %s
            """, (qr_path, book_id))
            self.db.commit()
            
            messagebox.showinfo("Success", f"Book '{data['title']}' added successfully!\nQR Code generated.")
            window.destroy()
            self.load_all_books()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add book: {str(e)}")
    
    def edit_book(self):
        """Edit selected book"""
        selected = self.manage_books_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a book to edit")
            return
        
        book_data = self.manage_books_tree.item(selected[0])['values']
        book_id = book_data[0]
        
        # Create edit window
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Book")
        edit_window.geometry("600x700")
        edit_window.configure(bg="#1a1a2e")
        edit_window.grab_set()
        
        # Header
        header = tk.Frame(edit_window, bg="#3498db")
        header.pack(fill="x")
        
        tk.Label(header, text="✏️ Edit Book", font=("Arial", 20, "bold"),
                bg="#3498db", fg="white").pack(pady=20)
        
        # Form
        form_frame = tk.Frame(edit_window, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        fields = [
            ("ISBN", book_data[1] if book_data[1] else ""),
            ("Title", book_data[2]),
            ("Author", book_data[3]),
            ("Category", book_data[4] if book_data[4] else ""),
            ("Quantity", str(book_data[5])),
            ("Location", book_data[7] if book_data[7] else "")
        ]
        
        entries = {}
        for label, value in fields:
            field_container = tk.Frame(form_frame, bg="#1a1a2e")
            field_container.pack(fill="x", pady=10)
            
            tk.Label(field_container, text=label, font=("Arial", 11, "bold"), 
                    bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
            
            entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                           bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                           relief="flat", bd=0)
            entry.pack(ipady=10)
            entry.insert(0, value)
            entries[label.lower()] = entry
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ UPDATE BOOK", font=("Arial", 12, "bold"),
                 bg="#3498db", fg="white", width=20, cursor="hand2",
                 relief="flat", bd=0,
                 command=lambda: self.update_book(book_id, entries, edit_window)).pack(side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                 bg="#95a5a6", fg="white", width=20, cursor="hand2",
                 relief="flat", bd=0,
                 command=edit_window.destroy).pack(side="left", padx=10, ipady=12)
    
    def update_book(self, book_id, entries, window):
        """Update book information"""
        data = {key: entry.get().strip() for key, entry in entries.items()}
        
        # Validate
        if not data['title'] or not data['author'] or not data['quantity']:
            messagebox.showwarning("Input Error", "Title, Author, and Quantity are required")
            return
        
        if not self.validate_book_title(data['title']):
            messagebox.showerror("Validation Error", "Invalid title format")
            return
        
        if not self.validate_name(data['author']):
            messagebox.showerror("Validation Error", "Author name should contain only letters and spaces")
            return
        
        if data['isbn'] and not self.validate_isbn(data['isbn']):
            messagebox.showerror("Validation Error", "Invalid ISBN format")
            return
        
        if not self.validate_number(data['quantity']):
            messagebox.showerror("Validation Error", "Quantity must be a positive integer")
            return
        
        try:
            # Get current available count
            self.cursor.execute("SELECT quantity, available FROM books WHERE book_id = %s", (book_id,))
            old_qty, old_avail = self.cursor.fetchone()
            
            new_qty = int(data['quantity'])
            # Adjust available based on quantity change
            borrowed = old_qty - old_avail
            new_avail = max(0, new_qty - borrowed)
            
            self.cursor.execute("""
                UPDATE books 
                SET isbn = %s, title = %s, author = %s, category = %s, 
                    quantity = %s, available = %s, location = %s
                WHERE book_id = %s
            """, (data['isbn'] if data['isbn'] else None,
                  data['title'],
                  data['author'],
                  data['category'] if data['category'] else None,
                  new_qty,
                  new_avail,
                  data['location'] if data['location'] else None,
                  book_id))
            
            self.db.commit()
            messagebox.showinfo("Success", "Book updated successfully!")
            window.destroy()
            self.load_all_books()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update book: {str(e)}")
    
    def delete_book(self):
        """Delete selected book"""
        selected = self.manage_books_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a book to delete")
            return
        
        book_data = self.manage_books_tree.item(selected[0])['values']
        book_id = book_data[0]
        title = book_data[2]
        
        # Check if book is currently borrowed
        self.cursor.execute("""
            SELECT COUNT(*) FROM transactions 
            WHERE book_id = %s AND status = 'borrowed'
        """, (book_id,))
        
        if self.cursor.fetchone()[0] > 0:
            messagebox.showerror("Error", "Cannot delete book that is currently borrowed")
            return
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete '{title}'?\n\nThis action cannot be undone."):
            try:
                # Delete book images
                self.cursor.execute("SELECT image_path, qr_code_path FROM books WHERE book_id = %s", (book_id,))
                result = self.cursor.fetchone()
                if result:
                    if result[0] and os.path.exists(result[0]):
                        os.remove(result[0])
                    if result[1] and os.path.exists(result[1]):
                        os.remove(result[1])
                
                # Delete transactions
                self.cursor.execute("DELETE FROM penalties WHERE transaction_id IN (SELECT transaction_id FROM transactions WHERE book_id = %s)", (book_id,))
                self.cursor.execute("DELETE FROM transactions WHERE book_id = %s", (book_id,))
                
                # Delete book
                self.cursor.execute("DELETE FROM books WHERE book_id = %s", (book_id,))
                self.db.commit()
                
                messagebox.showinfo("Success", f"Book '{title}' deleted successfully!")
                self.load_all_books()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete book: {str(e)}")
    
    def show_qr_code(self):
        """Display QR code for selected book"""
        selected = self.manage_books_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a book to view QR code")
            return
        
        book_data = self.manage_books_tree.item(selected[0])['values']
        book_id = book_data[0]
        title = book_data[2]
        
        # Get QR code path
        self.cursor.execute("SELECT qr_code_path FROM books WHERE book_id = %s", (book_id,))
        result = self.cursor.fetchone()
        
        if not result or not result[0] or not os.path.exists(result[0]):
            # Generate QR code if not exists
            qr_path = self.generate_qr_code(book_id, title)
            self.cursor.execute("UPDATE books SET qr_code_path = %s WHERE book_id = %s", (qr_path, book_id))
            self.db.commit()
        else:
            qr_path = result[0]
        
        # Display QR code
        qr_window = tk.Toplevel(self.root)
        qr_window.title(f"QR Code - {title}")
        qr_window.geometry("500x600")
        qr_window.configure(bg="#1a1a2e")
        qr_window.grab_set()
        
        header = tk.Frame(qr_window, bg="#9b59b6")
        header.pack(fill="x")
        
        tk.Label(header, text="📱 Book QR Code", font=("Arial", 20, "bold"),
                bg="#9b59b6", fg="white").pack(pady=20)
        
        content = tk.Frame(qr_window, bg="#1a1a2e")
        content.pack(fill="both", expand=True, padx=30, pady=30)
        
        tk.Label(content, text=title, font=("Arial", 14, "bold"),
                bg="#1a1a2e", fg="#ffffff", wraplength=400).pack(pady=10)
        
        tk.Label(content, text=f"Book ID: {book_id}", font=("Arial", 11),
                bg="#1a1a2e", fg="#a0a0a0").pack(pady=5)
        
        # QR code image
        img = Image.open(qr_path)
        img = img.resize((350, 350), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(content, image=photo, bg="#1a1a2e")
        img_label.image = photo
        img_label.pack(pady=20)
        
        tk.Button(content, text="CLOSE", font=("Arial", 12, "bold"),
                 bg="#95a5a6", fg="white", width=20, cursor="hand2",
                 relief="flat", bd=0, command=qr_window.destroy).pack(pady=20, ipady=12)
    
    def show_book_qr_code_student(self, book_id):
        """Display QR code for book (Student view)"""
        # Get book details
        self.cursor.execute("""
            SELECT title, author, category, qr_code_path 
            FROM books WHERE book_id = %s
        """, (book_id,))
        
        result = self.cursor.fetchone()
        
        if not result:
            messagebox.showerror("Error", "Book not found")
            return
        
        title, author, category, qr_path = result
        
        # Generate QR code if not exists
        if not qr_path or not os.path.exists(qr_path):
            qr_path = self.generate_qr_code(book_id, title)
            self.cursor.execute("""
                UPDATE books SET qr_code_path = %s WHERE book_id = %s
            """, (qr_path, book_id))
            self.db.commit()
        
        # Display QR code
        qr_window = tk.Toplevel(self.root)
        qr_window.title(f"QR Code - {title}")
        qr_window.geometry("500x650")
        qr_window.configure(bg="#1a1a2e")
        qr_window.grab_set()
        
        # Header (Student theme - red)
        header = tk.Frame(qr_window, bg="#e94560")
        header.pack(fill="x")
        
        tk.Label(header, text="📱 Book QR Code", font=("Arial", 20, "bold"),
                bg="#e94560", fg="white").pack(pady=20)
        
        # Content
        content = tk.Frame(qr_window, bg="#1a1a2e")
        content.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Book info
        tk.Label(content, text=title, font=("Arial", 14, "bold"),
                bg="#1a1a2e", fg="#ffffff", wraplength=400).pack(pady=10)
        
        tk.Label(content, text=f"by {author}", font=("Arial", 11, "italic"),
                bg="#1a1a2e", fg="#a0a0a0").pack(pady=5)
        
        if category:
            tk.Label(content, text=f"📖 {category}", font=("Arial", 10),
                    bg="#1a1a2e", fg="#7f8c8d").pack(pady=5)
        
        tk.Label(content, text=f"Book ID: {book_id}", font=("Arial", 11),
                bg="#1a1a2e", fg="#a0a0a0").pack(pady=5)
        
        # QR code image
        try:
            img = Image.open(qr_path)
            img = img.resize((350, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(content, image=photo, bg="#1a1a2e")
            img_label.image = photo
            img_label.pack(pady=20)
        except Exception as e:
            tk.Label(content, text=f"Error loading QR code: {str(e)}", 
                    font=("Arial", 10), bg="#1a1a2e", fg="#e74c3c").pack(pady=20)
        
        # Instructions
        info_frame = tk.Frame(content, bg="#0f3460", relief="flat", bd=0)
        info_frame.pack(fill="x", pady=10)
        
        info_content = tk.Frame(info_frame, bg="#0f3460")
        info_content.pack(padx=15, pady=15)
        
        tk.Label(info_content, text="📱 How to use:", font=("Arial", 11, "bold"),
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        tk.Label(info_content, text="Scan this QR code with your phone to get the Book ID",
                font=("Arial", 10), bg="#0f3460", fg="#a0a0a0", wraplength=400).pack(anchor="w")
        tk.Label(info_content, text="Use the Book ID for Quick Borrow or Quick Return",
                font=("Arial", 10), bg="#0f3460", fg="#a0a0a0", wraplength=400).pack(anchor="w")
        
        # Close button
        tk.Button(content, text="CLOSE", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0, command=qr_window.destroy).pack(pady=20, ipady=12)
    
    def view_borrowed_book_qr(self):
        """View QR code of selected borrowed book"""
        selected = self.borrowed_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a book to view its QR code")
            return
        
        trans_id = self.borrowed_tree.item(selected[0])['values'][0]
        
        # Get book_id from transaction
        self.cursor.execute("""
            SELECT book_id FROM transactions WHERE transaction_id = %s
        """, (trans_id,))
        
        result = self.cursor.fetchone()
        if result:
            self.show_book_qr_code_student(result[0])
        else:
            messagebox.showerror("Error", "Book not found")

    def show_manage_users(self):
        """Display user management interface with CRUD operations"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="👥 Manage Users", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Manage student and librarian accounts", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Action buttons
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Button(btn_frame, text="➕ ADD USER", font=("Arial", 11, "bold"),
                bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_add_user_form).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="✏️ EDIT USER", font=("Arial", 11, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.edit_user).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🗑️ DELETE USER", font=("Arial", 11, "bold"),
                bg="#e74c3c", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.delete_user).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.load_all_users).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        # Table frame
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        columns = ('ID', 'Username', 'Full Name', 'Email', 'Phone', 'Role', 'Created At')
        self.users_tree = ttk.Treeview(table_frame, columns=columns, 
                                    show='headings', height=15, style="Custom.Treeview")
        
        for col in columns:
            self.users_tree.heading(col, text=col)
            width = 200 if col in ['Full Name', 'Email'] else 120
            self.users_tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", 
                                command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=scrollbar.set)
        
        self.users_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        self.load_all_users()
    
    def load_all_users(self):
        """Load all users"""
        for item in self.users_tree.get_children():
            self.users_tree.delete(item)
        
        self.cursor.execute("""
            SELECT user_id, username, full_name, email, phone, role, 
                DATE_FORMAT(created_at, '%Y-%m-%d %H:%i')
            FROM users ORDER BY user_id DESC
        """)
        
        for row in self.cursor.fetchall():
            self.users_tree.insert('', 'end', values=row)
        
    def show_add_user_form(self):
        """Show add user form"""
        form_window = tk.Toplevel(self.root)
        form_window.title("Add New User")
        form_window.geometry("600x700")
        form_window.configure(bg="#1a1a2e")
        form_window.grab_set()
        
        # Header
        header = tk.Frame(form_window, bg="#27ae60")
        header.pack(fill="x")
        
        tk.Label(header, text="➕ Add New User", font=("Arial", 20, "bold"),
                bg="#27ae60", fg="white").pack(pady=20)
        
        # Form
        form_frame = tk.Frame(form_window, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        fields = [
            ("Username", "username", "4-20 characters"),
            ("Full Name", "full_name", "Full legal name"),
            ("Email", "email", "email@example.com"),
            ("Phone", "phone", "01XXXXXXXXX"),
            ("Password", "password", "Strong password", True)
        ]
        
        entries = {}
        for label, key, placeholder, *is_pass in fields:
            field_container = tk.Frame(form_frame, bg="#1a1a2e")
            field_container.pack(fill="x", pady=10)
            
            tk.Label(field_container, text=label + " *", font=("Arial", 11, "bold"), 
                    bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
            
            entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                           show="●" if is_pass else "",
                           bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                           relief="flat", bd=0)
            entry.pack(ipady=10)
            entry.insert(0, placeholder)
            entry.bind("<FocusIn>", lambda e, ent=entry, ph=placeholder: 
                      ent.delete(0, 'end') if ent.get() == ph else None)
            entries[key] = entry
        
        # Role selection
        role_container = tk.Frame(form_frame, bg="#1a1a2e")
        role_container.pack(fill="x", pady=10)
        
        tk.Label(role_container, text="Role *", font=("Arial", 11, "bold"), 
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        role_var = tk.StringVar(value="student")
        role_frame = tk.Frame(role_container, bg="#1a1a2e")
        role_frame.pack(anchor="w")
        
        tk.Radiobutton(role_frame, text="Student", variable=role_var, value="student",
                      font=("Arial", 11), bg="#1a1a2e", fg="#ffffff",
                      selectcolor="#0f3460", activebackground="#1a1a2e").pack(side="left", padx=10)
        tk.Radiobutton(role_frame, text="Librarian", variable=role_var, value="librarian",
                      font=("Arial", 11), bg="#1a1a2e", fg="#ffffff",
                      selectcolor="#0f3460", activebackground="#1a1a2e").pack(side="left", padx=10)
        
        entries['role'] = role_var
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ ADD USER", font=("Arial", 12, "bold"),
                 bg="#27ae60", fg="white", width=20, cursor="hand2",
                 relief="flat", bd=0,
                 command=lambda: self.add_user(entries, form_window)).pack(side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                 bg="#95a5a6", fg="white", width=20, cursor="hand2",
                 relief="flat", bd=0,
                 command=form_window.destroy).pack(side="left", padx=10, ipady=12)
    
    def add_user(self, entries, window):
        """Add new user with validation"""
        data = {}
        placeholders = ["4-20 characters", "Full legal name", "email@example.com", 
                       "01XXXXXXXXX", "Strong password", ""]
        
        for key, entry in entries.items():
            if key == 'role':
                data[key] = entry.get()
            else:
                value = entry.get().strip()
                if value in placeholders:
                    messagebox.showwarning("Input Error", f"Please fill in {key.replace('_', ' ')}")
                    return
                data[key] = value
        
        # Validate username
        if not self.validate_username(data['username']):
            messagebox.showerror("Validation Error", "Username must be 4-20 characters (letters, numbers, underscore)")
            return
        
        # Validate name
        if not self.validate_name(data['full_name']):
            messagebox.showerror("Validation Error", "Full name should contain only letters and spaces")
            return
        
        # Validate email
        if not self.validate_email(data['email']):
            messagebox.showerror("Validation Error", "Invalid email format")
            return
        
        # Validate phone
        if not self.validate_phone(data['phone']):
            messagebox.showerror("Validation Error", "Invalid phone number (Malaysian format)")
            return
        
        # Validate password
        is_valid, message = self.validate_password(data['password'])
        if not is_valid:
            messagebox.showerror("Validation Error", message)
            return
        
        # Check if username exists
        self.cursor.execute("SELECT * FROM users WHERE username = %s", (data['username'],))
        if self.cursor.fetchone():
            messagebox.showerror("Error", "Username already exists")
            return
        
        # Check if email exists
        self.cursor.execute("SELECT * FROM users WHERE email = %s", (data['email'],))
        if self.cursor.fetchone():
            messagebox.showerror("Error", "Email already registered")
            return
        
        try:
            hashed_pw = self.hash_password(data['password'])
            self.cursor.execute("""
                INSERT INTO users (username, password, full_name, email, phone, role)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (data['username'], hashed_pw, data['full_name'], 
                  data['email'], data['phone'], data['role']))
            self.db.commit()
            
            messagebox.showinfo("Success", f"User '{data['username']}' added successfully!")
            window.destroy()
            self.load_all_users()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add user: {str(e)}")
    
    def edit_user(self):
        """Edit selected user"""
        selected = self.users_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a user to edit")
            return
        
        user_data = self.users_tree.item(selected[0])['values']
        user_id = user_data[0]
        
        # Don't allow editing current logged-in user from here
        if user_id == self.current_user['user_id']:
            messagebox.showwarning("Warning", "You cannot edit your own account from here")
            return
        
        # Create edit window
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit User")
        edit_window.geometry("600x600")
        edit_window.configure(bg="#1a1a2e")
        edit_window.grab_set()
        
        # Header
        header = tk.Frame(edit_window, bg="#3498db")
        header.pack(fill="x")
        
        tk.Label(header, text="✏️ Edit User", font=("Arial", 20, "bold"),
                bg="#3498db", fg="white").pack(pady=20)
        
        # Form
        form_frame = tk.Frame(edit_window, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        fields = [
            ("Username", user_data[1]),
            ("Full Name", user_data[2]),
            ("Email", user_data[3]),
            ("Phone", user_data[4])
        ]
        
        entries = {}
        for label, value in fields:
            field_container = tk.Frame(form_frame, bg="#1a1a2e")
            field_container.pack(fill="x", pady=10)
            
            tk.Label(field_container, text=label, font=("Arial", 11, "bold"), 
                    bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
            
            entry = tk.Entry(field_container, font=("Arial", 11), width=50,
                           bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                           relief="flat", bd=0)
            entry.pack(ipady=10)
            entry.insert(0, value if value else "")
            
            # Username is not editable
            if label == "Username":
                entry.config(state="disabled")
            
            entries[label.lower().replace(" ", "_")] = entry
        
        # Role selection
        role_container = tk.Frame(form_frame, bg="#1a1a2e")
        role_container.pack(fill="x", pady=10)
        
        tk.Label(role_container, text="Role", font=("Arial", 11, "bold"), 
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        role_var = tk.StringVar(value=user_data[5])
        role_frame = tk.Frame(role_container, bg="#1a1a2e")
        role_frame.pack(anchor="w")
        
        tk.Radiobutton(role_frame, text="Student", variable=role_var, value="student",
                      font=("Arial", 11), bg="#1a1a2e", fg="#ffffff",
                      selectcolor="#0f3460", activebackground="#1a1a2e").pack(side="left", padx=10)
        tk.Radiobutton(role_frame, text="Librarian", variable=role_var, value="librarian",
                      font=("Arial", 11), bg="#1a1a2e", fg="#ffffff",
                      selectcolor="#0f3460", activebackground="#1a1a2e").pack(side="left", padx=10)
        
        entries['role'] = role_var
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ UPDATE USER", font=("Arial", 12, "bold"),
                 bg="#3498db", fg="white", width=20, cursor="hand2",
                 relief="flat", bd=0,
                 command=lambda: self.update_user(user_id, entries, edit_window)).pack(side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                 bg="#95a5a6", fg="white", width=20, cursor="hand2",
                 relief="flat", bd=0,
                 command=edit_window.destroy).pack(side="left", padx=10, ipady=12)
    
    def update_user(self, user_id, entries, window):
        """Update user information"""
        data = {}
        for key, entry in entries.items():
            if key == 'role':
                data[key] = entry.get()
            else:
                if entry.cget('state') == 'disabled':
                    continue
                data[key] = entry.get().strip()
        
        # Validate
        if not data['full_name'] or not data['email'] or not data['phone']:
            messagebox.showwarning("Input Error", "All fields are required")
            return
        
        if not self.validate_name(data['full_name']):
            messagebox.showerror("Validation Error", "Invalid name format")
            return
        
        if not self.validate_email(data['email']):
            messagebox.showerror("Validation Error", "Invalid email format")
            return
        
        if not self.validate_phone(data['phone']):
            messagebox.showerror("Validation Error", "Invalid phone number")
            return
        
        # Check if email exists for other users
        self.cursor.execute("SELECT user_id FROM users WHERE email = %s AND user_id != %s", 
                          (data['email'], user_id))
        if self.cursor.fetchone():
            messagebox.showerror("Error", "Email already registered to another user")
            return
        
        try:
            self.cursor.execute("""
                UPDATE users 
                SET full_name = %s, email = %s, phone = %s, role = %s
                WHERE user_id = %s
            """, (data['full_name'], data['email'], data['phone'], data['role'], user_id))
            
            self.db.commit()
            messagebox.showinfo("Success", "User updated successfully!")
            window.destroy()
            self.load_all_users()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update user: {str(e)}")
    
    def delete_user(self):
        """Delete selected user"""
        selected = self.users_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a user to delete")
            return
        
        user_data = self.users_tree.item(selected[0])['values']
        user_id = user_data[0]
        username = user_data[1]
        
        # Don't allow deleting current user
        if user_id == self.current_user['user_id']:
            messagebox.showerror("Error", "You cannot delete your own account")
            return
        
        # Check if user has active borrows
        self.cursor.execute("""
            SELECT COUNT(*) FROM transactions 
            WHERE user_id = %s AND status = 'borrowed'
        """, (user_id,))
        
        if self.cursor.fetchone()[0] > 0:
            messagebox.showerror("Error", "Cannot delete user with active borrowed books")
            return
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Deletion", 
                              f"Are you sure you want to delete user '{username}'?\n\nThis will also delete all their transaction history and penalties."):
            try:
                # Delete related records
                self.cursor.execute("DELETE FROM penalties WHERE user_id = %s", (user_id,))
                self.cursor.execute("DELETE FROM transactions WHERE user_id = %s", (user_id,))
                self.cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                self.db.commit()
                
                messagebox.showinfo("Success", f"User '{username}' deleted successfully!")
                self.load_all_users()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete user: {str(e)}")
    
    def show_transactions(self):
        """Display all transactions"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📊 Transaction History", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="View all book borrowing and return transactions", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Stats cards
        stats_frame = tk.Frame(self.content_frame, bg="#0f3460")
        stats_frame.pack(fill="x", padx=30, pady=10)
        
        # Get statistics
        self.cursor.execute("SELECT COUNT(*) FROM transactions WHERE status = 'borrowed'")
        active = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM transactions WHERE status = 'returned'")
        returned = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM transactions WHERE due_date < CURDATE() AND status = 'borrowed'")
        overdue = self.cursor.fetchone()[0]
        
        stats = [
            ("📖 Active Borrows", active, "#3498db"),
            ("✅ Returned", returned, "#27ae60"),
            ("⚠️ Overdue", overdue, "#e74c3c")
        ]
        
        for label, value, color in stats:
            card = tk.Frame(stats_frame, bg=color, width=250, height=100)
            card.pack(side="left", padx=10, pady=10)
            card.pack_propagate(False)
            
            tk.Label(card, text=label, font=("Arial", 12, "bold"), 
                    bg=color, fg="white").pack(pady=(20, 5))
            tk.Label(card, text=str(value), font=("Arial", 24, "bold"), 
                    bg=color, fg="white").pack()
        
        # Table frame
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        columns = ('Trans ID', 'User', 'Book', 'Borrow Date', 'Due Date', 'Return Date', 'Status')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', 
                           height=12, style="Custom.Treeview")
        
        for col in columns:
            tree.heading(col, text=col)
            width = 200 if col in ['User', 'Book'] else 120
            tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Load transactions
        self.cursor.execute("""
            SELECT t.transaction_id, u.full_name, b.title, 
                   t.borrow_date, t.due_date, t.return_date, t.status
            FROM transactions t
            JOIN users u ON t.user_id = u.user_id
            JOIN books b ON t.book_id = b.book_id
            ORDER BY t.transaction_id DESC
        """)
        
        for row in self.cursor.fetchall():
            tree.insert('', 'end', values=row)
    
    def show_reports(self):
        """Display reports and statistics"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📈 Reports & Analytics", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Library statistics and insights", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Create scrollable content
        canvas = tk.Canvas(self.content_frame, bg="#0f3460", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f3460")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Get statistics
        self.cursor.execute("SELECT COUNT(*) FROM books")
        total_books = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT SUM(quantity) FROM books")
        total_copies = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("SELECT SUM(available) FROM books")
        available_copies = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'student'")
        total_students = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM transactions WHERE status = 'borrowed'")
        active_borrows = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT SUM(amount) FROM penalties WHERE paid = 0")
        unpaid_penalties = self.cursor.fetchone()[0] or 0
        
        # Statistics cards
        stats_container = tk.Frame(scrollable_frame, bg="#0f3460")
        stats_container.pack(padx=30, pady=20, fill="x")
        
        stats = [
            ("📚 Total Books", total_books, "#3498db"),
            ("📖 Total Copies", total_copies, "#9b59b6"),
            ("✅ Available", available_copies, "#27ae60"),
            ("👥 Students", total_students, "#e67e22"),
            ("📊 Active Borrows", active_borrows, "#e74c3c"),
            ("💰 Unpaid Penalties", f"RM {unpaid_penalties:.2f}", "#c0392b")
        ]
        
        row_frame = None
        for idx, (label, value, color) in enumerate(stats):
            if idx % 3 == 0:
                row_frame = tk.Frame(stats_container, bg="#0f3460")
                row_frame.pack(fill="x", pady=10)
            
            card = tk.Frame(row_frame, bg=color, width=280, height=120)
            card.pack(side="left", padx=10)
            card.pack_propagate(False)
            
            tk.Label(card, text=label, font=("Arial", 13, "bold"), 
                    bg=color, fg="white").pack(pady=(25, 10))
            tk.Label(card, text=str(value), font=("Arial", 26, "bold"), 
                    bg=color, fg="white").pack()
            
        # Category Statistics Section
        tk.Label(scrollable_frame, text="📊 Category Distribution", 
                font=("Arial", 18, "bold"), bg="#0f3460", fg="#ffffff").pack(
                    anchor="w", padx=40, pady=(30, 10))
        
        category_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
        category_frame.pack(fill="x", padx=40, pady=10)
        
        self.cursor.execute("""
            SELECT 
                c.category_name,
                COUNT(b.book_id) as book_count,
                SUM(b.quantity) as total_copies
            FROM categories c
            LEFT JOIN books b ON c.category_name = b.category
            GROUP BY c.category_id
            ORDER BY book_count DESC
            LIMIT 10
        """)
        
        for cat_name, book_count, copies in self.cursor.fetchall():
            cat_card = tk.Frame(category_frame, bg="#0f3460")
            cat_card.pack(fill="x", pady=5, padx=10)
            
            tk.Label(cat_card, text=f"📂 {cat_name}", font=("Arial", 12, "bold"), 
                    bg="#0f3460", fg="white").pack(side="left", padx=10)
            tk.Label(cat_card, text=f"{book_count or 0} books | {copies or 0} copies", 
                    font=("Arial", 11), 
                    bg="#0f3460", fg="#a0a0a0").pack(side="right", padx=10)

        # Most borrowed books
        tk.Label(scrollable_frame, text="🔥 Most Borrowed Books (Top 5)", 
                font=("Arial", 18, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w", padx=40, pady=(30, 10))
        
        books_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
        books_frame.pack(fill="x", padx=40, pady=10)
        
        self.cursor.execute("""
            SELECT b.title, b.author, COUNT(t.transaction_id) as borrow_count
            FROM books b
            LEFT JOIN transactions t ON b.book_id = t.book_id
            GROUP BY b.book_id
            ORDER BY borrow_count DESC
            LIMIT 5
        """)
        
        for idx, (title, author, count) in enumerate(self.cursor.fetchall(), 1):
            book_card = tk.Frame(books_frame, bg="#0f3460")
            book_card.pack(fill="x", pady=5, padx=10)
            
            tk.Label(book_card, text=f"{idx}.", font=("Arial", 14, "bold"), 
                    bg="#0f3460", fg="#e94560").pack(side="left", padx=10)
            tk.Label(book_card, text=f"{title} by {author}", font=("Arial", 12), 
                    bg="#0f3460", fg="white").pack(side="left", padx=10)
            tk.Label(book_card, text=f"{count} borrows", font=("Arial", 11, "bold"), 
                    bg="#0f3460", fg="#27ae60").pack(side="right", padx=10)
        
        # Active borrowers
        tk.Label(scrollable_frame, text="👤 Most Active Borrowers (Top 5)", 
                font=("Arial", 18, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w", padx=40, pady=(30, 10))
        
        users_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
        users_frame.pack(fill="x", padx=40, pady=10)
        
        self.cursor.execute("""
            SELECT u.full_name, u.username, COUNT(t.transaction_id) as borrow_count
            FROM users u
            LEFT JOIN transactions t ON u.user_id = t.user_id
            WHERE u.role = 'student'
            GROUP BY u.user_id
            ORDER BY borrow_count DESC
            LIMIT 5
        """)
        
        for idx, (name, username, count) in enumerate(self.cursor.fetchall(), 1):
            user_card = tk.Frame(users_frame, bg="#0f3460")
            user_card.pack(fill="x", pady=5, padx=10)
            
            tk.Label(user_card, text=f"{idx}.", font=("Arial", 14, "bold"), 
                    bg="#0f3460", fg="#e94560").pack(side="left", padx=10)
            tk.Label(user_card, text=f"{name} (@{username})", font=("Arial", 12), 
                    bg="#0f3460", fg="white").pack(side="left", padx=10)
            tk.Label(user_card, text=f"{count} books", font=("Arial", 11, "bold"), 
                    bg="#0f3460", fg="#3498db").pack(side="right", padx=10)
        
        canvas.pack(side="left", fill="both", expand=True, padx=30, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)

    # 3. MAIN CATEGORY MANAGEMENT INTERFACE
    def show_manage_categories(self):
        """Display category management interface"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📑 Manage Categories", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Organize and manage book categories", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Action buttons
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Button(btn_frame, text="➕ ADD CATEGORY", font=("Arial", 11, "bold"),
                bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_add_category_form).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="✏️ EDIT CATEGORY", font=("Arial", 11, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.edit_category).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🗑️ DELETE CATEGORY", font=("Arial", 11, "bold"),
                bg="#e74c3c", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.delete_category).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="📊 VIEW STATISTICS", font=("Arial", 11, "bold"),
                bg="#9b59b6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_category_statistics).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.load_all_categories).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        # Table frame
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.Treeview", 
                    background="#1a1a2e",
                    foreground="white",
                    fieldbackground="#1a1a2e",
                    borderwidth=0)
        style.configure("Custom.Treeview.Heading",
                    background="#e94560",
                    foreground="white",
                    borderwidth=0)
        style.map("Custom.Treeview",
                background=[("selected", "#e94560")])
        
        columns = ('ID', 'Category Name', 'Description', 'Books Count', 'Created At')
        self.categories_tree = ttk.Treeview(table_frame, columns=columns, 
                                        show='headings', height=15, style="Custom.Treeview")
        
        for col in columns:
            self.categories_tree.heading(col, text=col)
            if col == 'Description':
                width = 350
            elif col == 'Category Name':
                width = 200
            else:
                width = 120
            self.categories_tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", 
                                command=self.categories_tree.yview)
        self.categories_tree.configure(yscrollcommand=scrollbar.set)
        
        self.categories_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        self.load_all_categories()


    # 4. LOAD CATEGORIES
    def load_all_categories(self):
        """Load all categories with book counts"""
        for item in self.categories_tree.get_children():
            self.categories_tree.delete(item)
        
        self.cursor.execute("""
            SELECT 
                c.category_id,
                c.category_name,
                c.description,
                COUNT(b.book_id) as book_count,
                DATE_FORMAT(c.created_at, '%Y-%m-%d %H:%i')
            FROM categories c
            LEFT JOIN books b ON c.category_name = b.category
            GROUP BY c.category_id
            ORDER BY c.category_name ASC
        """)
        
        for row in self.cursor.fetchall():
            self.categories_tree.insert('', 'end', values=row)


    # 5. ADD CATEGORY FORM
    def show_add_category_form(self):
        """Show add category form"""
        form_window = tk.Toplevel(self.root)
        form_window.title("Add New Category")
        form_window.geometry("600x500")
        form_window.configure(bg="#1a1a2e")
        form_window.grab_set()
        
        # Header
        header = tk.Frame(form_window, bg="#27ae60")
        header.pack(fill="x")
        
        tk.Label(header, text="➕ Add New Category", font=("Arial", 20, "bold"),
                bg="#27ae60", fg="white").pack(pady=20)
        
        # Form
        form_frame = tk.Frame(form_window, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Category Name
        tk.Label(form_frame, text="Category Name *", font=("Arial", 11, "bold"), 
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        name_entry = tk.Entry(form_frame, font=("Arial", 11), width=50,
                            bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        name_entry.pack(ipady=10, pady=(0, 20))
        name_entry.insert(0, "e.g., Science Fiction, Mathematics")
        name_entry.bind("<FocusIn>", lambda e: name_entry.delete(0, 'end') 
                    if name_entry.get().startswith("e.g.,") else None)
        
        # Description
        tk.Label(form_frame, text="Description", font=("Arial", 11, "bold"), 
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        desc_text = tk.Text(form_frame, font=("Arial", 11), width=50, height=6,
                        bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                        relief="flat", bd=0, wrap="word")
        desc_text.pack(pady=(0, 20))
        desc_text.insert("1.0", "Brief description of this category...")
        desc_text.bind("<FocusIn>", lambda e: desc_text.delete("1.0", 'end') 
                    if desc_text.get("1.0", 'end-1c').startswith("Brief") else None)
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ ADD CATEGORY", font=("Arial", 12, "bold"),
                bg="#27ae60", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.add_category(name_entry, desc_text, form_window)).pack(
                    side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=form_window.destroy).pack(side="left", padx=10, ipady=12)


    # 6. ADD CATEGORY
    def add_category(self, name_entry, desc_text, window):
        """Add new category with validation"""
        from tkinter import messagebox
        
        name = name_entry.get().strip()
        desc = desc_text.get("1.0", 'end-1c').strip()
        
        # Validation
        if not name or name.startswith("e.g.,"):
            messagebox.showwarning("Input Error", "Please enter a category name")
            return
        
        if len(name) < 2 or len(name) > 100:
            messagebox.showerror("Validation Error", "Category name must be 2-100 characters")
            return
        
        # Check valid characters (letters, numbers, spaces, hyphens)
        import re
        if not re.match(r'^[a-zA-Z0-9\s\-&]+$', name):
            messagebox.showerror("Validation Error", 
                            "Category name can only contain letters, numbers, spaces, hyphens, and &")
            return
        
        if desc.startswith("Brief"):
            desc = None
        
        # Check if category exists
        self.cursor.execute("SELECT * FROM categories WHERE category_name = %s", (name,))
        if self.cursor.fetchone():
            messagebox.showerror("Error", "Category already exists")
            return
        
        try:
            self.cursor.execute("""
                INSERT INTO categories (category_name, description, created_by)
                VALUES (%s, %s, %s)
            """, (name, desc, self.current_user['user_id']))
            self.db.commit()
            
            messagebox.showinfo("Success", f"Category '{name}' added successfully!")
            window.destroy()
            self.load_all_categories()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add category: {str(e)}")


    # 7. EDIT CATEGORY
    def edit_category(self):
        """Edit selected category"""
        from tkinter import messagebox
        
        selected = self.categories_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a category to edit")
            return
        
        cat_data = self.categories_tree.item(selected[0])['values']
        cat_id = cat_data[0]
        old_name = cat_data[1]
        old_desc = cat_data[2] if cat_data[2] else ""
        
        # Create edit window
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Category")
        edit_window.geometry("600x500")
        edit_window.configure(bg="#1a1a2e")
        edit_window.grab_set()
        
        # Header
        header = tk.Frame(edit_window, bg="#3498db")
        header.pack(fill="x")
        
        tk.Label(header, text="✏️ Edit Category", font=("Arial", 20, "bold"),
                bg="#3498db", fg="white").pack(pady=20)
        
        # Form
        form_frame = tk.Frame(edit_window, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Category Name
        tk.Label(form_frame, text="Category Name *", font=("Arial", 11, "bold"), 
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        name_entry = tk.Entry(form_frame, font=("Arial", 11), width=50,
                            bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        name_entry.pack(ipady=10, pady=(0, 20))
        name_entry.insert(0, old_name)
        
        # Description
        tk.Label(form_frame, text="Description", font=("Arial", 11, "bold"), 
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        desc_text = tk.Text(form_frame, font=("Arial", 11), width=50, height=6,
                        bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                        relief="flat", bd=0, wrap="word")
        desc_text.pack(pady=(0, 20))
        if old_desc:
            desc_text.insert("1.0", old_desc)
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ UPDATE CATEGORY", font=("Arial", 12, "bold"),
                bg="#3498db", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.update_category(cat_id, old_name, name_entry, 
                                                    desc_text, edit_window)).pack(
                    side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=edit_window.destroy).pack(side="left", padx=10, ipady=12)


    # 8. UPDATE CATEGORY
    def update_category(self, cat_id, old_name, name_entry, desc_text, window):
        """Update category information"""
        from tkinter import messagebox
        import re
        
        name = name_entry.get().strip()
        desc = desc_text.get("1.0", 'end-1c').strip()
        
        # Validation
        if not name:
            messagebox.showwarning("Input Error", "Category name cannot be empty")
            return
        
        if len(name) < 2 or len(name) > 100:
            messagebox.showerror("Validation Error", "Category name must be 2-100 characters")
            return
        
        if not re.match(r'^[a-zA-Z0-9\s\-&]+$', name):
            messagebox.showerror("Validation Error", 
                            "Category name can only contain letters, numbers, spaces, hyphens, and &")
            return
        
        if not desc:
            desc = None
        
        # Check if new name already exists (if name changed)
        if name != old_name:
            self.cursor.execute("SELECT * FROM categories WHERE category_name = %s", (name,))
            if self.cursor.fetchone():
                messagebox.showerror("Error", "Category name already exists")
                return
        
        try:
            # Update category
            self.cursor.execute("""
                UPDATE categories 
                SET category_name = %s, description = %s
                WHERE category_id = %s
            """, (name, desc, cat_id))
            
            # Update books with this category (if name changed)
            if name != old_name:
                self.cursor.execute("""
                    UPDATE books 
                    SET category = %s 
                    WHERE category = %s
                """, (name, old_name))
            
            self.db.commit()
            messagebox.showinfo("Success", "Category updated successfully!")
            window.destroy()
            self.load_all_categories()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update category: {str(e)}")


    # 9. DELETE CATEGORY
    def delete_category(self):
        """Delete selected category"""
        from tkinter import messagebox
        
        selected = self.categories_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a category to delete")
            return
        
        cat_data = self.categories_tree.item(selected[0])['values']
        cat_id = cat_data[0]
        cat_name = cat_data[1]
        book_count = cat_data[3]
        
        # Confirm deletion
        if book_count > 0:
            msg = (f"Category '{cat_name}' has {book_count} book(s).\n\n"
                f"Deleting this category will set all these books' category to NULL.\n\n"
                f"Are you sure you want to continue?")
        else:
            msg = f"Are you sure you want to delete category '{cat_name}'?"
        
        if messagebox.askyesno("Confirm Deletion", msg):
            try:
                # Set books' category to NULL
                self.cursor.execute("""
                    UPDATE books 
                    SET category = NULL 
                    WHERE category = %s
                """, (cat_name,))
                
                # Delete category
                self.cursor.execute("DELETE FROM categories WHERE category_id = %s", (cat_id,))
                self.db.commit()
                
                messagebox.showinfo("Success", f"Category '{cat_name}' deleted successfully!")
                self.load_all_categories()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete category: {str(e)}")


    # 10. CATEGORY STATISTICS
    def show_category_statistics(self):
        """Display category statistics in a new window"""
        from tkinter import messagebox
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Category Statistics")
        stats_window.geometry("900x700")
        stats_window.configure(bg="#1a1a2e")
        stats_window.grab_set()
        
        header = tk.Frame(stats_window, bg="#9b59b6")
        header.pack(fill="x")
        
        tk.Label(header, text="📊 Category Statistics", font=("Arial", 20, "bold"),
                bg="#9b59b6", fg="white").pack(pady=20)
        
        canvas = tk.Canvas(stats_window, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(stats_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=880)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Get statistics
        self.cursor.execute("""
            SELECT c.category_name, COUNT(b.book_id) as total_books,
                SUM(b.quantity) as total_copies, SUM(b.available) as available_copies,
                COUNT(DISTINCT t.transaction_id) as total_borrows
            FROM categories c
            LEFT JOIN books b ON c.category_name = b.category
            LEFT JOIN transactions t ON b.book_id = t.book_id
            GROUP BY c.category_id
            ORDER BY total_books DESC
        """)
        
        categories_stats = self.cursor.fetchall()
        
        if not categories_stats:
            tk.Label(scrollable_frame, text="No categories found", 
                    font=("Arial", 14), bg="#1a1a2e", fg="#a0a0a0").pack(pady=50)
        else:
            tk.Label(scrollable_frame, text="📚 Categories Overview", 
                    font=("Arial", 18, "bold"), bg="#1a1a2e", fg="#ffffff").pack(
                        anchor="w", padx=30, pady=(20, 10))
            
            for cat_name, books, copies, avail, borrows in categories_stats:
                card = tk.Frame(scrollable_frame, bg="#0f3460", relief="flat", bd=0)
                card.pack(fill="x", padx=30, pady=10)
                
                name_frame = tk.Frame(card, bg="#e94560")
                name_frame.pack(fill="x")
                tk.Label(name_frame, text=f"📂 {cat_name}", font=("Arial", 14, "bold"),
                        bg="#e94560", fg="white").pack(anchor="w", padx=20, pady=10)
                
                stats_frame = tk.Frame(card, bg="#0f3460")
                stats_frame.pack(fill="x", padx=20, pady=15)
                
                stats = [("📚 Books", books or 0), ("📖 Total Copies", copies or 0),
                        ("✅ Available", avail or 0), ("📊 Total Borrows", borrows or 0)]
                
                for label, value in stats:
                    stat_item = tk.Frame(stats_frame, bg="#0f3460")
                    stat_item.pack(side="left", padx=15)
                    tk.Label(stat_item, text=label, font=("Arial", 10),
                            bg="#0f3460", fg="#a0a0a0").pack()
                    tk.Label(stat_item, text=str(value), font=("Arial", 16, "bold"),
                            bg="#0f3460", fg="#ffffff").pack()
            
            tk.Label(scrollable_frame, text="🔥 Most Popular Categories (By Borrows)", 
                    font=("Arial", 18, "bold"), bg="#1a1a2e", fg="#ffffff").pack(
                        anchor="w", padx=30, pady=(30, 10))
            
            self.cursor.execute("""
                SELECT c.category_name, COUNT(t.transaction_id) as borrow_count
                FROM categories c
                LEFT JOIN books b ON c.category_name = b.category
                LEFT JOIN transactions t ON b.book_id = t.book_id
                GROUP BY c.category_id
                ORDER BY borrow_count DESC
                LIMIT 5
            """)
            
            popular_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
            popular_frame.pack(fill="x", padx=30, pady=10)
            
            for idx, (cat_name, count) in enumerate(self.cursor.fetchall(), 1):
                item = tk.Frame(popular_frame, bg="#0f3460")
                item.pack(fill="x", pady=5)
                tk.Label(item, text=f"{idx}.", font=("Arial", 14, "bold"),
                        bg="#0f3460", fg="#e94560").pack(side="left", padx=20)
                tk.Label(item, text=cat_name, font=("Arial", 12),
                        bg="#0f3460", fg="white").pack(side="left", padx=10)
                tk.Label(item, text=f"{count} borrows", font=("Arial", 11, "bold"),
                        bg="#0f3460", fg="#27ae60").pack(side="right", padx=20)
        
        tk.Button(scrollable_frame, text="CLOSE", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=30, cursor="hand2",
                relief="flat", bd=0, command=stats_window.destroy).pack(pady=30, ipady=12)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def show_browse_by_category(self):
        """Browse books by category"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📂 Browse by Category", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Explore books organized by categories", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # ============================================================
        # FETCH CATEGORIES FROM DATABASE - THIS WAS MISSING!
        # ============================================================
        self.cursor.execute("""
            SELECT 
                c.category_name,
                c.description,
                COUNT(b.book_id) as book_count
            FROM categories c
            LEFT JOIN books b ON c.category_name = b.category
            GROUP BY c.category_id
            HAVING book_count > 0
            ORDER BY c.category_name ASC
        """)
        
        categories = self.cursor.fetchall()
        # ============================================================
        
        # Scrollable content
        canvas = tk.Canvas(self.content_frame, bg="#0f3460", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f3460")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Check if categories exist
        if not categories:
            tk.Label(scrollable_frame, text="😔 No categories with books found", 
                    font=("Arial", 16), bg="#0f3460", fg="#a0a0a0").pack(pady=50)
            tk.Label(scrollable_frame, text="Books will appear here once categories are assigned", 
                    font=("Arial", 12), bg="#0f3460", fg="#7f8c8d").pack(pady=10)
        else:
            # Display categories as clickable cards
            for cat_name, cat_desc, book_count in categories:
                card = tk.Frame(scrollable_frame, bg="#1a1a2e", relief="flat", bd=0,
                            cursor="hand2")
                card.pack(fill="x", padx=30, pady=10)
                
                # Make card clickable
                card.bind("<Button-1>", lambda e, cn=cat_name: self.show_category_books(cn))
                
                content = tk.Frame(card, bg="#1a1a2e")
                content.pack(fill="both", expand=True, padx=20, pady=15)
                
                # Category header
                header = tk.Frame(content, bg="#1a1a2e")
                header.pack(fill="x")
                
                tk.Label(header, text=f"📂 {cat_name}", font=("Arial", 16, "bold"),
                        bg="#1a1a2e", fg="#e94560").pack(side="left")
                tk.Label(header, text=f"{book_count} book(s)", font=("Arial", 11, "bold"),
                        bg="#1a1a2e", fg="#27ae60").pack(side="right")
                
                # Description
                if cat_desc:
                    tk.Label(content, text=cat_desc, font=("Arial", 10),
                            bg="#1a1a2e", fg="#a0a0a0", wraplength=700,
                            justify="left").pack(anchor="w", pady=(5, 0))
                
                # Click hint
                tk.Label(content, text="👆 Click to view books in this category", 
                        font=("Arial", 9, "italic"),
                        bg="#1a1a2e", fg="#7f8c8d").pack(anchor="w", pady=(10, 0))
                
                # Bind all child widgets to make entire card clickable
                for widget in content.winfo_children():
                    widget.bind("<Button-1>", lambda e, cn=cat_name: self.show_category_books(cn))
                content.bind("<Button-1>", lambda e, cn=cat_name: self.show_category_books(cn))
        
        canvas.pack(side="left", fill="both", expand=True, padx=30, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
    # 13. SHOW BOOKS IN SELECTED CATEGORY
    def show_category_books(self, category_name):
        """Display all books in selected category"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        # Back button
        back_btn = tk.Button(header_frame, text="← Back to Categories", 
                            font=("Arial", 11, "bold"),
                            bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                            command=self.show_browse_by_category)
        back_btn.pack(anchor="w", pady=(0, 10), ipady=8, ipadx=15)
        
        tk.Label(header_frame, text=f"📂 {category_name}", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        
        # Get category description
        self.cursor.execute("SELECT description FROM categories WHERE category_name = %s", 
                        (category_name,))
        result = self.cursor.fetchone()
        if result and result[0]:
            tk.Label(header_frame, text=result[0], 
                    font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Results container
        results_container = tk.Frame(self.content_frame, bg="#0f3460")
        results_container.pack(fill="both", expand=True, padx=30, pady=10)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(results_container, bg="#0f3460", highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f3460")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Get books in this category
        self.cursor.execute("""
            SELECT book_id, title, author, category, available, location, image_path
            FROM books
            WHERE category = %s
            ORDER BY title ASC
        """, (category_name,))
        
        books = self.cursor.fetchall()
        
        if not books:
            tk.Label(scrollable_frame, text="😔 No books found in this category", 
                    font=("Arial", 16), bg="#0f3460", fg="#a0a0a0").pack(pady=50)
        else:
            tk.Label(scrollable_frame, text=f"Found {len(books)} book(s)", 
                    font=("Arial", 12, "bold"), bg="#0f3460", fg="#27ae60").pack(
                        anchor="w", padx=10, pady=10)
            
            # Display books in grid
            row_frame = None
            for idx, book in enumerate(books):
                if idx % 3 == 0:
                    row_frame = tk.Frame(scrollable_frame, bg="#0f3460")
                    row_frame.pack(fill="x", padx=10, pady=10)
                
                self.create_book_card(row_frame, book)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# 18. ENHANCED AI SEARCH WITH CATEGORY FILTER
    def ai_search_books_with_category(self, query, category):
        """AI-powered search with category filter"""
        # Base query
        if not query and category == "All Categories":
            self.cursor.execute("""
                SELECT book_id, title, author, category, available, location, image_path
                FROM books ORDER BY title ASC
            """)
            return self.cursor.fetchall()
        
        # Category filter only
        if not query and category != "All Categories":
            self.cursor.execute("""
                SELECT book_id, title, author, category, available, location, image_path
                FROM books WHERE category = %s ORDER BY title ASC
            """, (category,))
            return self.cursor.fetchall()
        
        # Search with category filter
        query_lower = query.lower()
        words = query_lower.split()
        
        # Build SQL with category filter
        sql = """
            SELECT DISTINCT b.book_id, b.title, b.author, b.category, 
                b.available, b.location, b.image_path,
                (
                    CASE 
                        WHEN LOWER(b.title) = %s THEN 100
                        WHEN LOWER(b.title) LIKE %s THEN 90
                        WHEN LOWER(b.author) = %s THEN 85
                        WHEN LOWER(b.author) LIKE %s THEN 75
                        WHEN LOWER(b.category) = %s THEN 70
                        WHEN LOWER(b.category) LIKE %s THEN 60
                        ELSE 50
                    END
                ) as relevance_score
            FROM books b
            WHERE (LOWER(b.title) LIKE %s 
            OR LOWER(b.author) LIKE %s 
            OR LOWER(b.category) LIKE %s
            OR LOWER(b.isbn) LIKE %s)
        """
        
        params = [
            query_lower, f'%{query_lower}%',
            query_lower, f'%{query_lower}%',
            query_lower, f'%{query_lower}%',
            f'%{query_lower}%', f'%{query_lower}%', 
            f'%{query_lower}%', f'%{query_lower}%'
        ]
        
        # Add category filter if not "All Categories"
        if category != "All Categories":
            sql += " AND b.category = %s"
            params.append(category)
        
        # Add word-by-word matching
        for word in words:
            if len(word) > 2:
                sql += " OR LOWER(b.title) LIKE %s OR LOWER(b.author) LIKE %s"
                params.extend([f'%{word}%', f'%{word}%'])
        
        sql += " ORDER BY relevance_score DESC, b.title ASC"
        
        self.cursor.execute(sql, params)
        return self.cursor.fetchall()

    def show_manage_penalties(self):
        """Display penalty management interface for librarians"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="💰 Manage Penalties", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Track and manage student penalties", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Summary cards
        summary_frame = tk.Frame(self.content_frame, bg="#0f3460")
        summary_frame.pack(fill="x", padx=30, pady=10)
        
        self.cursor.execute("SELECT SUM(amount) FROM penalties WHERE paid = 0")
        total_unpaid = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("SELECT SUM(amount) FROM penalties WHERE paid = 1")
        total_paid = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("SELECT COUNT(*) FROM penalties WHERE paid = 0")
        unpaid_count = self.cursor.fetchone()[0]
        
        stats = [
            ("💳 Total Unpaid", f"RM {total_unpaid:.2f}", "#e74c3c"),
            ("✅ Total Collected", f"RM {total_paid:.2f}", "#27ae60"),
            ("📊 Unpaid Count", str(unpaid_count), "#e67e22")
        ]
        
        for label, value, color in stats:
            card = tk.Frame(summary_frame, bg=color, width=280, height=100)
            card.pack(side="left", padx=10, pady=10)
            card.pack_propagate(False)
            
            tk.Label(card, text=label, font=("Arial", 13, "bold"), 
                    bg=color, fg="white").pack(pady=(20, 5))
            tk.Label(card, text=value, font=("Arial", 22, "bold"), 
                    bg=color, fg="white").pack(pady=(0, 20))
        
        # Action buttons
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Button(btn_frame, text="✓ MARK AS PAID", font=("Arial", 11, "bold"),
                bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.mark_penalty_paid).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="📄 PRINT RECEIPT", font=("Arial", 11, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.print_penalty_receipt).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_manage_penalties).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        # Table
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        columns = ('ID', 'Student Name', 'Days Overdue', 'Amount (RM)', 'Payment Date', 'Status')
        self.penalties_tree = ttk.Treeview(table_frame, columns=columns, show='headings', 
                                        height=12, style="Custom.Treeview")
        
        for col in columns:
            self.penalties_tree.heading(col, text=col)
            width = 200 if col == 'Student Name' else 120
            self.penalties_tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.penalties_tree.yview)
        self.penalties_tree.configure(yscrollcommand=scrollbar.set)
        
        self.penalties_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Load penalties
        self.cursor.execute("""
            SELECT p.penalty_id, u.full_name, p.days_overdue, p.amount, p.payment_date,
                CASE WHEN p.paid = 1 THEN 'Paid ✓' ELSE 'Unpaid ✗' END as status
            FROM penalties p
            JOIN users u ON p.user_id = u.user_id
            ORDER BY p.paid ASC, p.penalty_id DESC
        """)
        
        for row in self.cursor.fetchall():
            penalty_id, name, days, amount, pay_date, status = row
            display_date = pay_date if pay_date else "-"
            self.penalties_tree.insert('', 'end', values=(penalty_id, name, days, f"{amount:.2f}", display_date, status))


    # STEP 4: ADD - Mark Penalty as Paid
    # Add this NEW method:

    def mark_penalty_paid(self):
        """Mark selected penalty as paid"""
        selected = self.penalties_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a penalty to mark as paid")
            return
        
        penalty_data = self.penalties_tree.item(selected[0])['values']
        penalty_id = penalty_data[0]
        student_name = penalty_data[1]
        amount = penalty_data[3]
        status = penalty_data[5]
        
        if status == "Paid ✓":
            messagebox.showinfo("Info", "This penalty has already been paid")
            return
        
        # Confirm payment
        if messagebox.askyesno("Confirm Payment", 
                            f"Mark penalty as PAID?\n\nStudent: {student_name}\nAmount: RM {amount}\n\nThis action cannot be undone."):
            try:
                from datetime import datetime
                payment_date = datetime.now().date()
                
                self.cursor.execute("""
                    UPDATE penalties 
                    SET paid = 1, payment_date = %s
                    WHERE penalty_id = %s
                """, (payment_date, penalty_id))
                
                self.db.commit()
                messagebox.showinfo("Success", f"Penalty marked as PAID!\n\nReceipt Date: {payment_date}")
                self.show_manage_penalties()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update penalty: {str(e)}")


    # STEP 5: ADD - Print Receipt
    # Add this NEW method:

    def print_penalty_receipt(self):
        """Generate and display penalty receipt"""
        selected = self.penalties_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a penalty to print receipt")
            return
        
        penalty_data = self.penalties_tree.item(selected[0])['values']
        penalty_id = penalty_data[0]
        
        # Get full penalty details
        self.cursor.execute("""
            SELECT p.penalty_id, u.full_name, u.username, p.days_overdue, 
                p.amount, p.payment_date, p.paid, b.title, t.due_date, t.return_date
            FROM penalties p
            JOIN users u ON p.user_id = u.user_id
            JOIN transactions t ON p.transaction_id = t.transaction_id
            JOIN books b ON t.book_id = b.book_id
            WHERE p.penalty_id = %s
        """, (penalty_id,))
        
        result = self.cursor.fetchone()
        if not result:
            messagebox.showerror("Error", "Penalty details not found")
            return
        
        pen_id, name, username, days, amount, pay_date, paid, book, due, ret = result
        
        # Create receipt window
        receipt_window = tk.Toplevel(self.root)
        receipt_window.title(f"Penalty Receipt #{pen_id}")
        receipt_window.geometry("600x700")
        receipt_window.configure(bg="#ffffff")
        receipt_window.grab_set()
        
        # Receipt content
        receipt_frame = tk.Frame(receipt_window, bg="#ffffff", padx=40, pady=30)
        receipt_frame.pack(fill="both", expand=True)
        
        # Header
        tk.Label(receipt_frame, text="📚 LibraAI", font=("Arial", 28, "bold"),
                bg="#ffffff", fg="#e94560").pack(pady=(0, 5))
        tk.Label(receipt_frame, text="PENALTY RECEIPT", font=("Arial", 20, "bold"),
                bg="#ffffff", fg="#1a1a2e").pack(pady=(0, 5))
        tk.Label(receipt_frame, text="─" * 50, font=("Arial", 10),
                bg="#ffffff", fg="#cccccc").pack(pady=10)
        
        # Receipt details
        from datetime import datetime
        receipt_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        details = [
            ("Receipt ID:", f"#{pen_id}"),
            ("Date:", receipt_date),
            ("", ""),
            ("Student Name:", name),
            ("Username:", username),
            ("", ""),
            ("Book Title:", book),
            ("Due Date:", str(due)),
            ("Return Date:", str(ret)),
            ("Days Overdue:", f"{days} days"),
            ("", ""),
            ("Penalty Rate:", "RM 1.00 per day"),
            ("Total Amount:", f"RM {amount:.2f}"),
            ("", ""),
            ("Status:", "PAID ✓" if paid else "UNPAID ✗"),
            ("Payment Date:", str(pay_date) if pay_date else "-")
        ]
        
        for label, value in details:
            if not label and not value:
                tk.Label(receipt_frame, text="", bg="#ffffff").pack(pady=3)
                continue
            
            row = tk.Frame(receipt_frame, bg="#ffffff")
            row.pack(fill="x", pady=3)
            
            tk.Label(row, text=label, font=("Arial", 11, "bold" if label in ["Total Amount:", "Status:"] else "normal"),
                    bg="#ffffff", fg="#1a1a2e").pack(side="left")
            tk.Label(row, text=value, font=("Arial", 11, "bold" if label in ["Total Amount:", "Status:"] else "normal"),
                    bg="#ffffff", fg="#e94560" if label == "Total Amount:" else "#27ae60" if paid else "#e74c3c").pack(side="right")
        
        tk.Label(receipt_frame, text="─" * 50, font=("Arial", 10),
                bg="#ffffff", fg="#cccccc").pack(pady=20)
        
        tk.Label(receipt_frame, text="Thank you for your payment", font=("Arial", 10, "italic"),
                bg="#ffffff", fg="#7f8c8d").pack(pady=5)
        tk.Label(receipt_frame, text="LibraAI - AI-Powered Library Management System", 
                font=("Arial", 9), bg="#ffffff", fg="#a0a0a0").pack(pady=5)
        
        # Close button
        tk.Button(receipt_frame, text="CLOSE", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=30, cursor="hand2",
                relief="flat", bd=0, command=receipt_window.destroy).pack(pady=20, ipady=12)

    def show_student_profile(self):
        """Display student profile with statistics and edit options"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="👤 My Profile", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Manage your account and view statistics", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Scrollable content
        canvas = tk.Canvas(self.content_frame, bg="#0f3460", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f3460")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Get user data
        self.cursor.execute("""
            SELECT username, full_name, email, phone, created_at 
            FROM users WHERE user_id = %s
        """, (self.current_user['user_id'],))
        
        user_data = self.cursor.fetchone()
        username, full_name, email, phone, created_at = user_data
        
        # Profile Card
        profile_card = tk.Frame(scrollable_frame, bg="#1a1a2e")
        profile_card.pack(fill="x", padx=30, pady=20)
        
        # Profile header
        profile_header = tk.Frame(profile_card, bg="#e94560")
        profile_header.pack(fill="x")
        
        tk.Label(profile_header, text="👤 ACCOUNT INFORMATION", font=("Arial", 16, "bold"),
                bg="#e94560", fg="white").pack(anchor="w", padx=20, pady=15)
        
        # Profile details
        details_frame = tk.Frame(profile_card, bg="#1a1a2e")
        details_frame.pack(fill="x", padx=30, pady=20)
        
        profile_details = [
            ("👤 Full Name:", full_name),
            ("🆔 Username:", username),
            ("📧 Email:", email),
            ("📱 Phone:", phone),
            ("📅 Member Since:", created_at.strftime("%B %d, %Y") if created_at else "-")
        ]
        
        for label, value in profile_details:
            row = tk.Frame(details_frame, bg="#1a1a2e")
            row.pack(fill="x", pady=8)
            
            tk.Label(row, text=label, font=("Arial", 12, "bold"),
                    bg="#1a1a2e", fg="#a0a0a0", width=20, anchor="w").pack(side="left")
            tk.Label(row, text=value, font=("Arial", 12),
                    bg="#1a1a2e", fg="#ffffff").pack(side="left", padx=10)
        
        # Action buttons
        btn_frame = tk.Frame(profile_card, bg="#1a1a2e")
        btn_frame.pack(fill="x", padx=30, pady=(10, 20))
        
        tk.Button(btn_frame, text="✏️ EDIT PROFILE", font=("Arial", 11, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.edit_student_profile).pack(side="left", padx=5, ipady=10, ipadx=20)
        
        tk.Button(btn_frame, text="🔒 CHANGE PASSWORD", font=("Arial", 11, "bold"),
                bg="#9b59b6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.change_student_password).pack(side="left", padx=5, ipady=10, ipadx=20)
        
        # Statistics Section
        tk.Label(scrollable_frame, text="📊 MY STATISTICS", font=("Arial", 18, "bold"),
                bg="#0f3460", fg="#ffffff").pack(anchor="w", padx=40, pady=(20, 10))
        
        # Get statistics
        self.cursor.execute("""
            SELECT COUNT(*) FROM transactions WHERE user_id = %s
        """, (self.current_user['user_id'],))
        total_borrows = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM transactions WHERE user_id = %s AND status = 'borrowed'
        """, (self.current_user['user_id'],))
        active_borrows = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM transactions WHERE user_id = %s AND status = 'returned'
        """, (self.current_user['user_id'],))
        returned_books = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT SUM(amount) FROM penalties WHERE user_id = %s AND paid = 0
        """, (self.current_user['user_id'],))
        unpaid_penalties = self.cursor.fetchone()[0] or 0
        
        # Stats cards
        stats_container = tk.Frame(scrollable_frame, bg="#0f3460")
        stats_container.pack(fill="x", padx=30, pady=10)
        
        stats = [
            ("📚 Total Borrows", total_borrows, "#3498db"),
            ("📖 Active Borrows", active_borrows, "#e67e22"),
            ("✅ Returned Books", returned_books, "#27ae60"),
            ("💰 Unpaid Penalties", f"RM {unpaid_penalties:.2f}", "#e74c3c")
        ]
        
        row_frame = None
        for idx, (label, value, color) in enumerate(stats):
            if idx % 2 == 0:
                row_frame = tk.Frame(stats_container, bg="#0f3460")
                row_frame.pack(fill="x", pady=10)
            
            card = tk.Frame(row_frame, bg=color, width=380, height=120)
            card.pack(side="left", padx=10)
            card.pack_propagate(False)
            
            tk.Label(card, text=label, font=("Arial", 13, "bold"),
                    bg=color, fg="white").pack(pady=(25, 10))
            tk.Label(card, text=str(value), font=("Arial", 26, "bold"),
                    bg=color, fg="white").pack()
        
        canvas.pack(side="left", fill="both", expand=True, padx=30, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)

    def edit_student_profile(self):
        """Edit student profile information"""
        # Get current user data
        self.cursor.execute("""
            SELECT full_name, email, phone 
            FROM users WHERE user_id = %s
        """, (self.current_user['user_id'],))
        
        user_data = self.cursor.fetchone()
        current_name, current_email, current_phone = user_data
        
        # Create edit window
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Profile")
        edit_window.geometry("600x500")
        edit_window.configure(bg="#1a1a2e")
        edit_window.grab_set()
        
        # Header
        header = tk.Frame(edit_window, bg="#3498db")
        header.pack(fill="x")
        
        tk.Label(header, text="✏️ Edit Profile", font=("Arial", 20, "bold"),
                bg="#3498db", fg="white").pack(pady=20)
        
        # Form
        form_frame = tk.Frame(edit_window, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Full Name
        tk.Label(form_frame, text="Full Name *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        name_entry = tk.Entry(form_frame, font=("Arial", 11), width=50,
                            bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        name_entry.pack(ipady=10, pady=(0, 20))
        name_entry.insert(0, current_name)
        
        # Email
        tk.Label(form_frame, text="Email *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        email_entry = tk.Entry(form_frame, font=("Arial", 11), width=50,
                            bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        email_entry.pack(ipady=10, pady=(0, 20))
        email_entry.insert(0, current_email)
        
        # Phone
        tk.Label(form_frame, text="Phone *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        phone_entry = tk.Entry(form_frame, font=("Arial", 11), width=50,
                            bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        phone_entry.pack(ipady=10, pady=(0, 20))
        phone_entry.insert(0, current_phone)
        
        # Info message
        tk.Label(form_frame, text="Note: Username cannot be changed", 
                font=("Arial", 9, "italic"), bg="#1a1a2e", fg="#7f8c8d").pack(pady=10)
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ SAVE CHANGES", font=("Arial", 12, "bold"),
                bg="#27ae60", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.update_student_profile(
                    name_entry, email_entry, phone_entry, edit_window)).pack(
                        side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=edit_window.destroy).pack(side="left", padx=10, ipady=12)


    # STEP 4: ADD NEW METHOD - Update Profile
    # Add this NEW method:

    def update_student_profile(self, name_entry, email_entry, phone_entry, window):
        """Update student profile in database"""
        name = name_entry.get().strip()
        email = email_entry.get().strip()
        phone = phone_entry.get().strip()
        
        # Validation
        if not name or not email or not phone:
            messagebox.showwarning("Input Error", "All fields are required")
            return
        
        if not self.validate_name(name):
            messagebox.showerror("Validation Error", "Full name should contain only letters and spaces")
            return
        
        if not self.validate_email(email):
            messagebox.showerror("Validation Error", "Invalid email format")
            return
        
        if not self.validate_phone(phone):
            messagebox.showerror("Validation Error", "Invalid phone number (use Malaysian format: 01XXXXXXXXX)")
            return
        
        # Check if email exists for other users
        self.cursor.execute("""
            SELECT user_id FROM users WHERE email = %s AND user_id != %s
        """, (email, self.current_user['user_id']))
        
        if self.cursor.fetchone():
            messagebox.showerror("Error", "Email already registered to another user")
            return
        
        # Update profile
        try:
            self.cursor.execute("""
                UPDATE users 
                SET full_name = %s, email = %s, phone = %s
                WHERE user_id = %s
            """, (name, email, phone, self.current_user['user_id']))
            
            self.db.commit()
            
            # Update current_user session
            self.current_user['full_name'] = name
            
            messagebox.showinfo("Success", "Profile updated successfully!")
            window.destroy()
            self.show_student_profile()  # Refresh profile view
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update profile: {str(e)}")


    # STEP 5: ADD NEW METHOD - Change Password
    # Add this NEW method:

    def change_student_password(self):
        """Change student password"""
        # Create password change window
        pwd_window = tk.Toplevel(self.root)
        pwd_window.title("Change Password")
        pwd_window.geometry("600x550")
        pwd_window.configure(bg="#1a1a2e")
        pwd_window.grab_set()
        
        # Header
        header = tk.Frame(pwd_window, bg="#9b59b6")
        header.pack(fill="x")
        
        tk.Label(header, text="🔒 Change Password", font=("Arial", 20, "bold"),
                bg="#9b59b6", fg="white").pack(pady=20)
        
        # Form
        form_frame = tk.Frame(pwd_window, bg="#1a1a2e")
        form_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Current Password
        tk.Label(form_frame, text="Current Password *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        current_pwd_entry = tk.Entry(form_frame, font=("Arial", 11), width=50, show="●",
                                    bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                                    relief="flat", bd=0)
        current_pwd_entry.pack(ipady=10, pady=(0, 20))
        
        # New Password
        tk.Label(form_frame, text="New Password *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        new_pwd_entry = tk.Entry(form_frame, font=("Arial", 11), width=50, show="●",
                                bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                                relief="flat", bd=0)
        new_pwd_entry.pack(ipady=10, pady=(0, 10))
        
        # Password strength indicator
        strength_label = tk.Label(form_frame, text="", font=("Arial", 9),
                                bg="#1a1a2e")
        strength_label.pack(pady=5)
        
        new_pwd_entry.bind('<KeyRelease>', lambda e: self.check_password_strength(
            new_pwd_entry.get(), strength_label))
        
        # Confirm New Password
        tk.Label(form_frame, text="Confirm New Password *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(10, 5))
        
        confirm_pwd_entry = tk.Entry(form_frame, font=("Arial", 11), width=50, show="●",
                                    bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                                    relief="flat", bd=0)
        confirm_pwd_entry.pack(ipady=10, pady=(0, 20))
        
        # Password requirements
        requirements = tk.Frame(form_frame, bg="#0f3460", relief="flat", bd=0)
        requirements.pack(fill="x", pady=10, padx=10)
        
        tk.Label(requirements, text="Password Requirements:", font=("Arial", 10, "bold"),
                bg="#0f3460", fg="#ffffff").pack(anchor="w", padx=15, pady=(10, 5))
        
        reqs = [
            "• At least 8 characters long",
            "• Contains uppercase letter (A-Z)",
            "• Contains lowercase letter (a-z)",
            "• Contains number (0-9)",
            "• Contains special character (@$!%*?&#)"
        ]
        
        for req in reqs:
            tk.Label(requirements, text=req, font=("Arial", 9),
                    bg="#0f3460", fg="#a0a0a0").pack(anchor="w", padx=30, pady=2)
        
        # Buttons
        btn_container = tk.Frame(form_frame, bg="#1a1a2e")
        btn_container.pack(pady=30)
        
        tk.Button(btn_container, text="✓ CHANGE PASSWORD", font=("Arial", 12, "bold"),
                bg="#27ae60", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.update_student_password(
                    current_pwd_entry, new_pwd_entry, confirm_pwd_entry, pwd_window)).pack(
                        side="left", padx=10, ipady=12)
        
        tk.Button(btn_container, text="✗ CANCEL", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=pwd_window.destroy).pack(side="left", padx=10, ipady=12)


    # STEP 6: ADD NEW METHOD - Update Password
    # Add this NEW method:

    def update_student_password(self, current_pwd_entry, new_pwd_entry, confirm_pwd_entry, window):
        """Update student password in database"""
        current_pwd = current_pwd_entry.get().strip()
        new_pwd = new_pwd_entry.get().strip()
        confirm_pwd = confirm_pwd_entry.get().strip()
        
        # Validation
        if not current_pwd or not new_pwd or not confirm_pwd:
            messagebox.showwarning("Input Error", "All fields are required")
            return
        
        # Verify current password
        hashed_current = self.hash_password(current_pwd)
        self.cursor.execute("""
            SELECT password FROM users WHERE user_id = %s
        """, (self.current_user['user_id'],))
        
        stored_password = self.cursor.fetchone()[0]
        
        if hashed_current != stored_password:
            messagebox.showerror("Error", "Current password is incorrect")
            return
        
        # Validate new password
        is_valid, message = self.validate_password(new_pwd)
        if not is_valid:
            messagebox.showerror("Validation Error", message)
            return
        
        # Check password match
        if new_pwd != confirm_pwd:
            messagebox.showerror("Error", "New passwords do not match")
            return
        
        # Check if new password is same as current
        if current_pwd == new_pwd:
            messagebox.showwarning("Warning", "New password must be different from current password")
            return
        
        # Update password
        try:
            hashed_new = self.hash_password(new_pwd)
            self.cursor.execute("""
                UPDATE users 
                SET password = %s
                WHERE user_id = %s
            """, (hashed_new, self.current_user['user_id']))
            
            self.db.commit()
            
            messagebox.showinfo("Success", 
                            "Password changed successfully!\n\nPlease login again with your new password.")
            window.destroy()
            self.show_login()  # Force re-login
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to change password: {str(e)}")

    def generate_notifications(self):
        """Generate notifications for due/overdue books"""
        from datetime import datetime, timedelta
        
        self.check_available_reservations()
        
        today = datetime.now().date()
        
        # Check for books due in 2 days
        due_soon_date = today + timedelta(days=2)
        self.cursor.execute("""
            SELECT t.user_id, b.title, t.due_date, t.transaction_id, u.email, u.full_name
            FROM transactions t
            JOIN books b ON t.book_id = b.book_id
            JOIN users u ON t.user_id = u.user_id
            WHERE t.status = 'borrowed' 
            AND t.due_date BETWEEN CURDATE() AND %s
        """, (due_soon_date,))
        
        for user_id, title, due_date, trans_id, email, full_name in self.cursor.fetchall():
            # Check if notification already exists
            self.cursor.execute("""
                SELECT notification_id FROM notifications 
                WHERE user_id = %s AND title LIKE %s AND type = 'due_soon'
                AND created_at >= DATE_SUB(CURDATE(), INTERVAL 3 DAY)
            """, (user_id, f"%{title}%"))
            
            if not self.cursor.fetchone():
                # Create notification
                self.cursor.execute("""
                    INSERT INTO notifications (user_id, title, message, type)
                    VALUES (%s, %s, %s, 'due_soon')
                """, (user_id, 
                    f"Book Due Soon: {title}",
                    f"Your book '{title}' is due on {due_date}. Please return it on time to avoid penalties."))
                
                # SEND EMAIL (NEW)
                if email:
                    self.send_due_date_reminder_email(email, full_name, title, due_date)
        
        # Check for overdue books
        self.cursor.execute("""
            SELECT t.user_id, b.title, t.due_date, t.transaction_id,
                DATEDIFF(CURDATE(), t.due_date) as days, u.email, u.full_name
            FROM transactions t
            JOIN books b ON t.book_id = b.book_id
            JOIN users u ON t.user_id = u.user_id
            WHERE t.status = 'borrowed' 
            AND t.due_date < CURDATE()
        """)
        
        for user_id, title, due_date, trans_id, days, email, full_name in self.cursor.fetchall():
            # Check if notification already exists for today
            self.cursor.execute("""
                SELECT notification_id FROM notifications 
                WHERE user_id = %s AND title LIKE %s AND type = 'overdue'
                AND created_at >= DATE_SUB(CURDATE(), INTERVAL 3 DAY)
            """, (user_id, f"%{title}%"))
            
            if not self.cursor.fetchone():
                penalty = days * 1.00
                
                # Create notification
                self.cursor.execute("""
                    INSERT INTO notifications (user_id, title, message, type)
                    VALUES (%s, %s, %s, 'overdue')
                """, (user_id,
                    f"⚠️ OVERDUE: {title}",
                    f"Your book '{title}' is {days} day(s) overdue! Current penalty: RM {penalty:.2f}. Please return immediately."))
                
                # SEND EMAIL (NEW)
                if email:
                    self.send_overdue_notification_email(email, full_name, title, days, penalty)
        
        self.db.commit()

    def show_notifications(self):
        """Display notification center"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="🔔 Notifications", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Stay updated with your library activities", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Action buttons
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Button(btn_frame, text="✓ MARK ALL AS READ", font=("Arial", 11, "bold"),
                bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.mark_all_notifications_read).pack(side="left", padx=5, ipady=10, ipadx=20)
        
        tk.Button(btn_frame, text="🗑️ CLEAR ALL", font=("Arial", 11, "bold"),
                bg="#e74c3c", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.clear_all_notifications).pack(side="left", padx=5, ipady=10, ipadx=20)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=lambda: [self.generate_notifications(), self.show_notifications()]).pack(
                    side="left", padx=5, ipady=10, ipadx=20)
        
        # Notifications container
        canvas = tk.Canvas(self.content_frame, bg="#0f3460", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f3460")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Get notifications
        self.cursor.execute("""
            SELECT notification_id, title, message, type, is_read, created_at
            FROM notifications
            WHERE user_id = %s
            ORDER BY is_read ASC, created_at DESC
        """, (self.current_user['user_id'],))
        
        notifications = self.cursor.fetchall()
        
        if not notifications:
            tk.Label(scrollable_frame, text="📭 No notifications yet", 
                    font=("Arial", 16), bg="#0f3460", fg="#a0a0a0").pack(pady=50)
            tk.Label(scrollable_frame, text="You'll be notified about due dates and important updates", 
                    font=("Arial", 11), bg="#0f3460", fg="#7f8c8d").pack(pady=10)
        else:
            for notif_id, title, message, ntype, is_read, created_at in notifications:
                # Notification card
                card_bg = "#1a1a2e" if not is_read else "#0f3460"
                card = tk.Frame(scrollable_frame, bg=card_bg, relief="flat", bd=0)
                card.pack(fill="x", padx=30, pady=5)
                
                content = tk.Frame(card, bg=card_bg)
                content.pack(fill="x", padx=20, pady=15)
                
                # Icon and header
                header_frame = tk.Frame(content, bg=card_bg)
                header_frame.pack(fill="x")
                
                icons = {
                    'overdue': '⚠️',
                    'due_soon': '⏰',
                    'penalty': '💰',
                    'returned': '✅',
                    'info': 'ℹ️'
                }
                icon = icons.get(ntype, 'ℹ️')
                
                tk.Label(header_frame, text=icon, font=("Arial", 16),
                        bg=card_bg, fg="white").pack(side="left", padx=(0, 10))
                
                title_frame = tk.Frame(header_frame, bg=card_bg)
                title_frame.pack(side="left", fill="x", expand=True)
                
                tk.Label(title_frame, text=title, font=("Arial", 12, "bold"),
                        bg=card_bg, fg="#ffffff").pack(anchor="w")
                
                time_str = created_at.strftime("%B %d, %Y at %I:%M %p")
                tk.Label(title_frame, text=time_str, font=("Arial", 9),
                        bg=card_bg, fg="#7f8c8d").pack(anchor="w")
                
                if not is_read:
                    tk.Label(header_frame, text="●", font=("Arial", 20),
                            bg=card_bg, fg="#e74c3c").pack(side="right")
                
                # Message
                tk.Label(content, text=message, font=("Arial", 11),
                        bg=card_bg, fg="#a0a0a0", wraplength=700, justify="left").pack(
                            anchor="w", pady=(10, 0), padx=(26, 0))
                
                # Mark as read button
                if not is_read:
                    tk.Button(content, text="Mark as read", font=("Arial", 9),
                            bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                            command=lambda nid=notif_id: self.mark_notification_read(nid)).pack(
                                anchor="w", pady=(10, 0), padx=(26, 0), ipady=5, ipadx=10)
        
        canvas.pack(side="left", fill="both", expand=True, padx=30, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)


    # STEP 5: ADD NOTIFICATION POPUP ON LOGIN
    # Add this NEW method:

    def show_notification_popup(self):
        """Show popup for critical notifications on login"""
        # Check for critical notifications (overdue or due soon)
        self.cursor.execute("""
            SELECT title, message, type
            FROM notifications
            WHERE user_id = %s AND is_read = 0
            AND type IN ('overdue', 'due_soon')
            ORDER BY FIELD(type, 'overdue', 'due_soon'), created_at DESC
            LIMIT 3
        """, (self.current_user['user_id'],))
        
        critical_notifs = self.cursor.fetchall()
        
        if critical_notifs:
            popup = tk.Toplevel(self.root)
            popup.title("Important Notifications")
            popup.geometry("600x500")
            popup.configure(bg="#1a1a2e")
            popup.grab_set()
            
            # Header
            header = tk.Frame(popup, bg="#e74c3c" if any(n[2] == 'overdue' for n in critical_notifs) else "#f39c12")
            header.pack(fill="x")
            
            tk.Label(header, text="🔔 Important Notifications", font=("Arial", 20, "bold"),
                    bg=header['bg'], fg="white").pack(pady=20)
            
            # Content
            content = tk.Frame(popup, bg="#1a1a2e")
            content.pack(fill="both", expand=True, padx=30, pady=20)
            
            tk.Label(content, text="You have important library notifications:", 
                    font=("Arial", 12), bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 20))
            
            for title, message, ntype in critical_notifs:
                notif_frame = tk.Frame(content, bg="#0f3460", relief="flat", bd=0)
                notif_frame.pack(fill="x", pady=10)
                
                notif_content = tk.Frame(notif_frame, bg="#0f3460")
                notif_content.pack(fill="x", padx=15, pady=15)
                
                icon = "⚠️" if ntype == 'overdue' else "⏰"
                tk.Label(notif_content, text=icon, font=("Arial", 16),
                        bg="#0f3460", fg="white").pack(side="left", padx=(0, 10))
                
                text_frame = tk.Frame(notif_content, bg="#0f3460")
                text_frame.pack(side="left", fill="x", expand=True)
                
                tk.Label(text_frame, text=title, font=("Arial", 11, "bold"),
                        bg="#0f3460", fg="#ffffff").pack(anchor="w")
                tk.Label(text_frame, text=message, font=("Arial", 10),
                        bg="#0f3460", fg="#a0a0a0", wraplength=400, justify="left").pack(anchor="w")
            
            # Buttons
            btn_frame = tk.Frame(content, bg="#1a1a2e")
            btn_frame.pack(pady=30)
            
            tk.Button(btn_frame, text="VIEW ALL NOTIFICATIONS", font=("Arial", 11, "bold"),
                    bg="#3498db", fg="white", width=25, cursor="hand2",
                    relief="flat", bd=0,
                    command=lambda: [popup.destroy(), self.show_notifications()]).pack(
                        side="left", padx=5, ipady=12)
            
            tk.Button(btn_frame, text="DISMISS", font=("Arial", 11, "bold"),
                    bg="#95a5a6", fg="white", width=15, cursor="hand2",
                    relief="flat", bd=0,
                    command=popup.destroy).pack(side="left", padx=5, ipady=12)


    # STEP 6: ADD HELPER METHODS
    # Add these NEW methods:

    def mark_notification_read(self, notif_id):
        """Mark single notification as read"""
        try:
            self.cursor.execute("""
                UPDATE notifications SET is_read = 1 WHERE notification_id = %s
            """, (notif_id,))
            self.db.commit()
            self.show_notifications()  # Refresh
        except Exception as e:
            messagebox.showerror("Error", f"Failed to mark notification: {str(e)}")

    def mark_all_notifications_read(self):
        """Mark all notifications as read"""
        try:
            self.cursor.execute("""
                UPDATE notifications SET is_read = 1 WHERE user_id = %s
            """, (self.current_user['user_id'],))
            self.db.commit()
            messagebox.showinfo("Success", "All notifications marked as read")
            self.show_notifications()  # Refresh
        except Exception as e:
            messagebox.showerror("Error", f"Failed to mark notifications: {str(e)}")

    def clear_all_notifications(self):
        """Delete all notifications"""
        if messagebox.askyesno("Confirm", "Clear all notifications?\n\nThis cannot be undone."):
            try:
                self.cursor.execute("""
                    DELETE FROM notifications WHERE user_id = %s
                """, (self.current_user['user_id'],))
                self.db.commit()
                messagebox.showinfo("Success", "All notifications cleared")
                self.show_notifications()  # Refresh
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear notifications: {str(e)}")

    def reserve_book(self, book_id):
        """Reserve a book that's currently unavailable"""
        # Get book details
        self.cursor.execute("SELECT title, available FROM books WHERE book_id = %s", (book_id,))
        result = self.cursor.fetchone()
        
        if not result:
            messagebox.showerror("Error", "Book not found")
            return
        
        title, available = result
        
        if available > 0:
            messagebox.showinfo("Info", "This book is available now! You can borrow it directly.")
            return
        
        # Check if already reserved
        self.cursor.execute("""
            SELECT reservation_id FROM reservations 
            WHERE user_id = %s AND book_id = %s AND status = 'pending'
        """, (self.current_user['user_id'], book_id))
        
        if self.cursor.fetchone():
            messagebox.showinfo("Info", "You have already reserved this book")
            return
        
        # Check if user already has active borrows of this book
        self.cursor.execute("""
            SELECT transaction_id FROM transactions 
            WHERE user_id = %s AND book_id = %s AND status = 'borrowed'
        """, (self.current_user['user_id'], book_id))
        
        if self.cursor.fetchone():
            messagebox.showerror("Error", "You already have this book borrowed")
            return
        
        # Get queue position
        self.cursor.execute("""
            SELECT COUNT(*) FROM reservations 
            WHERE book_id = %s AND status = 'pending'
        """, (book_id,))
        queue_position = self.cursor.fetchone()[0] + 1
        
        # Create reservation
        try:
            self.cursor.execute("""
                INSERT INTO reservations (user_id, book_id, status)
                VALUES (%s, %s, 'pending')
            """, (self.current_user['user_id'], book_id))
            
            self.db.commit()
            
            messagebox.showinfo("Success", 
                f"Book reserved successfully!\n\n"
                f"Book: {title}\n"
                f"Queue Position: #{queue_position}\n\n"
                f"You'll be notified when the book becomes available.")
            
            # Refresh current view
            if hasattr(self, 'results_container'):
                # If in search results
                self.display_search_results_with_filter("", "All Categories")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reserve book: {str(e)}")


    # STEP 4: ADD MY RESERVATIONS PAGE
    # Add this NEW method:

    def show_my_reservations(self):
        """Display student's book reservations"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📋 My Reservations", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Track your reserved books and queue status", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Stats
        stats_frame = tk.Frame(self.content_frame, bg="#0f3460")
        stats_frame.pack(fill="x", padx=30, pady=10)
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM reservations 
            WHERE user_id = %s AND status = 'pending'
        """, (self.current_user['user_id'],))
        pending = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM reservations 
            WHERE user_id = %s AND status = 'available'
        """, (self.current_user['user_id'],))
        available = self.cursor.fetchone()[0]
        
        stats = [
            ("⏳ Pending", pending, "#f39c12"),
            ("✅ Available", available, "#27ae60")
        ]
        
        for label, value, color in stats:
            card = tk.Frame(stats_frame, bg=color, width=350, height=100)
            card.pack(side="left", padx=10, pady=10)
            card.pack_propagate(False)
            
            tk.Label(card, text=label, font=("Arial", 13, "bold"),
                    bg=color, fg="white").pack(pady=(20, 5))
            tk.Label(card, text=str(value), font=("Arial", 26, "bold"),
                    bg=color, fg="white").pack()
        
        # Action buttons
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Button(btn_frame, text="✗ CANCEL RESERVATION", font=("Arial", 11, "bold"),
                bg="#e74c3c", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.cancel_reservation).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_my_reservations).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        # Table
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        columns = ('Res ID', 'Book Title', 'Author', 'Queue Position', 'Reserved Date', 'Status')
        self.reservations_tree = ttk.Treeview(table_frame, columns=columns, 
                                            show='headings', height=12, style="Custom.Treeview")
        
        for col in columns:
            self.reservations_tree.heading(col, text=col)
            width = 250 if col == 'Book Title' else 150
            self.reservations_tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.reservations_tree.yview)
        self.reservations_tree.configure(yscrollcommand=scrollbar.set)
        
        self.reservations_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Load reservations
        self.cursor.execute("""
            SELECT r.reservation_id, b.title, b.author, r.reserved_at, r.status, r.book_id
            FROM reservations r
            JOIN books b ON r.book_id = b.book_id
            WHERE r.user_id = %s
            ORDER BY r.reserved_at DESC
        """, (self.current_user['user_id'],))
        
        for res_id, title, author, reserved_at, status, book_id in self.cursor.fetchall():
            # Calculate queue position
            if status == 'pending':
                self.cursor.execute("""
                    SELECT COUNT(*) + 1 FROM reservations 
                    WHERE book_id = %s AND status = 'pending' 
                    AND reserved_at < %s
                """, (book_id, reserved_at))
                queue_pos = f"#{self.cursor.fetchone()[0]}"
            else:
                queue_pos = "-"
            
            status_display = {
                'pending': '⏳ Pending',
                'available': '✅ Available',
                'cancelled': '✗ Cancelled',
                'fulfilled': '✓ Fulfilled'
            }.get(status, status)
            
            self.reservations_tree.insert('', 'end', values=(
                res_id, title, author, queue_pos, 
                reserved_at.strftime("%Y-%m-%d %H:%M"), status_display
            ))


    # STEP 5: ADD CANCEL RESERVATION METHOD
    # Add this NEW method:

    def cancel_reservation(self):
        """Cancel selected reservation"""
        selected = self.reservations_tree.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select a reservation to cancel")
            return
        
        res_data = self.reservations_tree.item(selected[0])['values']
        res_id = res_data[0]
        book_title = res_data[1]
        status = res_data[5]
        
        if 'Cancelled' in status or 'Fulfilled' in status:
            messagebox.showinfo("Info", "This reservation cannot be cancelled")
            return
        
        if messagebox.askyesno("Confirm Cancellation", 
                            f"Cancel reservation for '{book_title}'?\n\nYou can reserve it again later."):
            try:
                self.cursor.execute("""
                    UPDATE reservations 
                    SET status = 'cancelled'
                    WHERE reservation_id = %s
                """, (res_id,))
                
                self.db.commit()
                messagebox.showinfo("Success", "Reservation cancelled")
                self.show_my_reservations()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to cancel reservation: {str(e)}")


    # STEP 6: ADD AUTO-NOTIFICATION FOR AVAILABLE BOOKS
    # Add this NEW method:

    def check_available_reservations(self):
        """Check for books that became available and notify reserved users"""
        from datetime import datetime, timedelta
        
        # Get all pending reservations for books that now have copies available
        self.cursor.execute("""
            SELECT r.reservation_id, r.user_id, r.book_id, b.title, b.available
            FROM reservations r
            JOIN books b ON r.book_id = b.book_id
            WHERE r.status = 'pending' AND b.available > 0
            ORDER BY r.reserved_at ASC
        """)
        
        for res_id, user_id, book_id, title, available in self.cursor.fetchall():
            # Check if this is the first in queue
            self.cursor.execute("""
                SELECT COUNT(*) FROM reservations 
                WHERE book_id = %s AND status = 'pending' 
                AND reserved_at < (SELECT reserved_at FROM reservations WHERE reservation_id = %s)
            """, (book_id, res_id))
            
            queue_before = self.cursor.fetchone()[0]
            
            # If first in queue and book available
            if queue_before == 0:
                # Update reservation status
                expires_at = datetime.now() + timedelta(days=2)  # 2 days to claim
                
                self.cursor.execute("""
                    UPDATE reservations 
                    SET status = 'available', notified_at = NOW(), expires_at = %s
                    WHERE reservation_id = %s
                """, (expires_at, res_id))
                
                # Create notification
                self.cursor.execute("""
                    INSERT INTO notifications (user_id, title, message, type)
                    VALUES (%s, %s, %s, 'info')
                """, (user_id,
                    f"📚 Reserved Book Available: {title}",
                    f"Good news! Your reserved book '{title}' is now available. Please borrow it within 2 days or your reservation will expire."))
                
                # SEND EMAIL (NEW)
                self.cursor.execute("SELECT email, full_name FROM users WHERE user_id = %s", (user_id,))
                user_email, user_name = self.cursor.fetchone()
                if user_email:
                    self.send_reservation_available_email(user_email, user_name, title, expires_at)
                
                self.db.commit()
        
        # Check for expired reservations
        self.cursor.execute("""
            UPDATE reservations 
            SET status = 'cancelled'
            WHERE status = 'available' AND expires_at < NOW()
        """)
        self.db.commit()

    def show_manage_reservations(self):
        """Display all reservations for librarian"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="📋 Manage Reservations", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="View and manage all book reservations", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Stats
        stats_frame = tk.Frame(self.content_frame, bg="#0f3460")
        stats_frame.pack(fill="x", padx=30, pady=10)
        
        self.cursor.execute("SELECT COUNT(*) FROM reservations WHERE status = 'pending'")
        pending = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM reservations WHERE status = 'available'")
        available = self.cursor.fetchone()[0]
        
        stats = [
            ("⏳ Pending", pending, "#f39c12"),
            ("✅ Available to Claim", available, "#27ae60")
        ]
        
        for label, value, color in stats:
            card = tk.Frame(stats_frame, bg=color, width=380, height=100)
            card.pack(side="left", padx=10, pady=10)
            card.pack_propagate(False)
            
            tk.Label(card, text=label, font=("Arial", 13, "bold"),
                    bg=color, fg="white").pack(pady=(20, 5))
            tk.Label(card, text=str(value), font=("Arial", 26, "bold"),
                    bg=color, fg="white").pack()
        
        # Buttons
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Button(btn_frame, text="🔄 CHECK AVAILABILITY", font=("Arial", 11, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=lambda: [self.check_available_reservations(), 
                                self.show_manage_reservations()]).pack(
                                    side="left", padx=5, ipady=12, ipadx=20)
        
        tk.Button(btn_frame, text="🔄 REFRESH", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_manage_reservations).pack(side="left", padx=5, ipady=12, ipadx=20)
        
        # Table
        table_frame = tk.Frame(self.content_frame, bg="#1a1a2e")
        table_frame.pack(fill="both", expand=True, padx=30, pady=10)
        
        columns = ('Res ID', 'Student', 'Book Title', 'Reserved Date', 'Expires', 'Status')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', 
                        height=12, style="Custom.Treeview")
        
        for col in columns:
            tree.heading(col, text=col)
            width = 200 if col in ['Student', 'Book Title'] else 120
            tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Load reservations
        self.cursor.execute("""
            SELECT r.reservation_id, u.full_name, b.title, r.reserved_at, 
                r.expires_at, r.status
            FROM reservations r
            JOIN users u ON r.user_id = u.user_id
            JOIN books b ON r.book_id = b.book_id
            ORDER BY r.status ASC, r.reserved_at ASC
        """)
        
        for res_id, student, title, reserved, expires, status in self.cursor.fetchall():
            expires_str = expires.strftime("%Y-%m-%d %H:%M") if expires else "-"
            
            status_display = {
                'pending': '⏳ Pending',
                'available': '✅ Available',
                'cancelled': '✗ Cancelled',
                'fulfilled': '✓ Fulfilled'
            }.get(status, status)
            
            tree.insert('', 'end', values=(
                res_id, student, title, 
                reserved.strftime("%Y-%m-%d %H:%M"), 
                expires_str, status_display
            ))

    def show_recommendations(self):
        """Display AI-powered book recommendations"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.content_frame, bg="#0f3460")
        header_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Label(header_frame, text="🤖 AI Book Recommendations", 
                font=("Arial", 24, "bold"), bg="#0f3460", fg="#ffffff").pack(anchor="w")
        tk.Label(header_frame, text="Personalized suggestions powered by Genetic Algorithm", 
                font=("Arial", 11), bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=5)
        
        # Info banner
        info_frame = tk.Frame(self.content_frame, bg="#3498db")
        info_frame.pack(fill="x", padx=30, pady=10)
        
        info_content = tk.Frame(info_frame, bg="#3498db")
        info_content.pack(fill="x", padx=20, pady=15)
        
        tk.Label(info_content, text="💡", font=("Arial", 20),
                bg="#3498db", fg="white").pack(side="left", padx=10)
        
        text_frame = tk.Frame(info_content, bg="#3498db")
        text_frame.pack(side="left", fill="x", expand=True)
        
        tk.Label(text_frame, text="How it works:", font=("Arial", 11, "bold"),
                bg="#3498db", fg="white").pack(anchor="w")
        tk.Label(text_frame, 
                text="Our AI analyzes your borrowing history, preferences, and reading patterns to suggest books you'll love!",
                font=("Arial", 10), bg="#3498db", fg="white", wraplength=600, justify="left").pack(anchor="w")
        
        # Generate button
        btn_frame = tk.Frame(self.content_frame, bg="#0f3460")
        btn_frame.pack(fill="x", padx=30, pady=20)
        
        tk.Button(btn_frame, text="🧬 GENERATE RECOMMENDATIONS", font=("Arial", 12, "bold"),
                bg="#e94560", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.generate_ai_recommendations).pack(ipady=15, ipadx=30)
        
        tk.Label(btn_frame, text="Using Genetic Algorithm with 30 generations", 
                font=("Arial", 9, "italic"), bg="#0f3460", fg="#7f8c8d").pack(pady=5)
        
        tk.Button(btn_frame, text="ℹ️ HOW IT WORKS", font=("Arial", 10, "bold"),
                bg="#3498db", fg="white", cursor="hand2", relief="flat", bd=0,
                command=self.show_ga_explanation).pack(pady=10, ipady=10, ipadx=20)
        
        # Recommendations container
        self.recommendations_container = tk.Frame(self.content_frame, bg="#0f3460")
        self.recommendations_container.pack(fill="both", expand=True, padx=30, pady=10)
        
        # Show initial message
        tk.Label(self.recommendations_container, 
                text="👆 Click the button above to get personalized recommendations", 
                font=("Arial", 14), bg="#0f3460", fg="#a0a0a0").pack(pady=100)


    # STEP 3: ADD RECOMMENDATION GENERATION METHOD
    # Add this NEW method:

    def generate_ai_recommendations(self):
        """Generate recommendations using Genetic Algorithm"""
        # Clear container
        for widget in self.recommendations_container.winfo_children():
            widget.destroy()
        
        # Show loading
        loading_label = tk.Label(self.recommendations_container, 
                                text="🧬 Genetic Algorithm Running...\nAnalyzing your preferences...", 
                                font=("Arial", 14, "bold"), bg="#0f3460", fg="#ffffff")
        loading_label.pack(pady=50)
        
        self.root.update()
        
        # Initialize GA engine
        ga_engine = GeneticRecommendationEngine(
            population_size=20,
            generations=30,
            mutation_rate=0.1
        )
        
        # Generate recommendations
        try:
            recommendations = ga_engine.generate_recommendations(
                self.cursor, 
                self.current_user['user_id'],
                num_recommendations=6
            )
            
            # ADD THIS DEBUG LINE:
            print(f"DEBUG: Got {len(recommendations)} recommendations")
            
            # Clear loading
            loading_label.destroy()
            
            if not recommendations:
                tk.Label(self.recommendations_container, 
                        text="😔 No recommendations available", 
                        font=("Arial", 16), bg="#0f3460", fg="#a0a0a0").pack(pady=50)
                tk.Label(self.recommendations_container, 
                        text="Try borrowing some books first to help our AI learn your preferences!", 
                        font=("Arial", 11), bg="#0f3460", fg="#7f8c8d").pack(pady=10)
                return
            
            # Display recommendations
            tk.Label(self.recommendations_container, 
                    text=f"✨ Top {len(recommendations)} Recommendations for You", 
                    font=("Arial", 16, "bold"), bg="#0f3460", fg="#27ae60").pack(anchor="w", padx=10, pady=10)
            
            # Create scrollable frame
            canvas = tk.Canvas(self.recommendations_container, bg="#0f3460", highlightthickness=0)
            scrollbar = ttk.Scrollbar(self.recommendations_container, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg="#0f3460")
            
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Display recommendation cards
            row_frame = None
            for idx, book in enumerate(recommendations):
                if idx % 3 == 0:
                    row_frame = tk.Frame(scrollable_frame, bg="#0f3460")
                    row_frame.pack(fill="x", padx=10, pady=10)
                
                self.create_recommendation_card(row_frame, book, idx + 1)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            loading_label.destroy()
            # IMPROVED ERROR MESSAGE:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR: {error_details}")
            
            tk.Label(self.recommendations_container, 
                    text=f"❌ Error: {str(e)}", 
                    font=("Arial", 12), bg="#0f3460", fg="#e74c3c").pack(pady=20)
            tk.Label(self.recommendations_container, 
                    text="Check console for details", 
                    font=("Arial", 10), bg="#0f3460", fg="#7f8c8d").pack(pady=5)


    # STEP 4: ADD RECOMMENDATION CARD CREATOR
    # Add this NEW method:

    def create_recommendation_card(self, parent, book, rank):
        """Create a recommendation card with ranking"""
        card = tk.Frame(parent, bg="#1a1a2e", width=280, height=420, relief="flat", bd=0)
        card.pack(side="left", padx=10, pady=10)
        card.pack_propagate(False)
        
        # Rank badge
        rank_colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
        rank_color = rank_colors.get(rank, "#3498db")
        
        rank_badge = tk.Frame(card, bg=rank_color, height=40)
        rank_badge.pack(fill="x")
        
        tk.Label(rank_badge, text=f"#{rank} RECOMMENDED", font=("Arial", 10, "bold"),
                bg=rank_color, fg="#1a1a2e").pack(pady=10)
        
        # Book image placeholder
        img_frame = tk.Frame(card, bg="#0f3460", width=260, height=180)
        img_frame.pack(pady=10, padx=10)
        img_frame.pack_propagate(False)
        
        # Try to load image if exists
        self.cursor.execute("SELECT image_path FROM books WHERE book_id = %s", (book['book_id'],))
        result = self.cursor.fetchone()
        image_path = result[0] if result else None
        
        try:
            if image_path and os.path.exists(image_path):
                from PIL import Image, ImageTk
                img = Image.open(image_path)
                img = img.resize((260, 180), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = tk.Label(img_frame, image=photo, bg="#0f3460")
                img_label.image = photo
                img_label.pack()
            else:
                tk.Label(img_frame, text="📚", font=("Arial", 60), 
                        bg="#0f3460", fg="#e94560").pack(expand=True)
        except:
            tk.Label(img_frame, text="📚", font=("Arial", 60), 
                    bg="#0f3460", fg="#e94560").pack(expand=True)
        
        # Book info
        info_frame = tk.Frame(card, bg="#1a1a2e")
        info_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Title
        title_text = book['title'] if len(book['title']) <= 30 else book['title'][:27] + "..."
        tk.Label(info_frame, text=title_text, font=("Arial", 12, "bold"), 
                bg="#1a1a2e", fg="#ffffff", wraplength=250).pack(anchor="w")
        
        # Author
        tk.Label(info_frame, text=f"by {book['author']}", font=("Arial", 10, "italic"), 
                bg="#1a1a2e", fg="#a0a0a0").pack(anchor="w", pady=2)
        
        # Category
        if book['category']:
            tk.Label(info_frame, text=f"📖 {book['category']}", font=("Arial", 9), 
                    bg="#1a1a2e", fg="#7f8c8d").pack(anchor="w", pady=2)
        
        # Popularity indicator
        popularity = book['popularity']
        pop_text = "🔥 Trending" if popularity > 5 else "⭐ Popular" if popularity > 2 else "✨ Hidden Gem"
        tk.Label(info_frame, text=pop_text, font=("Arial", 9, "bold"), 
                bg="#1a1a2e", fg="#f39c12").pack(anchor="w", pady=5)
        
        # Borrow button
        tk.Button(info_frame, text="BORROW NOW", font=("Arial", 10, "bold"),
                bg="#27ae60", fg="white", cursor="hand2", relief="flat", bd=0,
                command=lambda: self.borrow_book_by_id(book['book_id'])).pack(fill="x", pady=5, ipady=8)

    # STEP 6: ADD GA EXPLANATION PAGE (OPTIONAL - FOR DEMO)
    # Add this NEW method:

    def show_ga_explanation(self):
        """Show how the Genetic Algorithm works"""
        explanation_window = tk.Toplevel(self.root)
        explanation_window.title("How GA Recommendations Work")
        explanation_window.geometry("700x600")
        explanation_window.configure(bg="#1a1a2e")
        explanation_window.grab_set()
        
        # Header
        header = tk.Frame(explanation_window, bg="#9b59b6")
        header.pack(fill="x")
        
        tk.Label(header, text="🧬 Genetic Algorithm Explained", font=("Arial", 20, "bold"),
                bg="#9b59b6", fg="white").pack(pady=20)
        
        # Content
        canvas = tk.Canvas(explanation_window, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(explanation_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        content = tk.Frame(scrollable_frame, bg="#1a1a2e")
        content.pack(fill="both", expand=True, padx=40, pady=30)
        
        sections = [
            ("🎯 What is it?", 
            "Our recommendation system uses a Genetic Algorithm (GA) - an AI technique inspired by natural evolution. It evolves book recommendations over multiple generations to find the best matches for you."),
            
            ("🧬 How it works:",
            "1. INITIALIZATION: Creates 20 random book combinations\n"
            "2. FITNESS EVALUATION: Scores each combination based on:\n"
            "   • Your category preferences (40%)\n"
            "   • Similar users' choices (30%)\n"
            "   • Book popularity (20%)\n"
            "   • Variety/diversity (10%)\n"
            "3. SELECTION: Chooses best combinations\n"
            "4. CROSSOVER: Combines good solutions\n"
            "5. MUTATION: Adds random changes (10% chance)\n"
            "6. REPEAT: Evolves for 30 generations"),
            
            ("⚡ Why it's powerful:",
            "• Learns from YOUR unique reading patterns\n"
            "• Considers what similar readers enjoyed\n"
            "• Balances popularity with hidden gems\n"
            "• Continuously improves recommendations\n"
            "• Handles complex preference patterns"),
            
            ("📊 The Result:",
            "After 30 generations of evolution, the algorithm produces a highly optimized list of books tailored specifically to your tastes!")
        ]
        
        for title, text in sections:
            section_frame = tk.Frame(content, bg="#0f3460", relief="flat", bd=0)
            section_frame.pack(fill="x", pady=10)
            
            section_content = tk.Frame(section_frame, bg="#0f3460")
            section_content.pack(fill="x", padx=20, pady=15)
            
            tk.Label(section_content, text=title, font=("Arial", 14, "bold"),
                    bg="#0f3460", fg="#e94560").pack(anchor="w", pady=(0, 10))
            
            tk.Label(section_content, text=text, font=("Arial", 11),
                    bg="#0f3460", fg="#ffffff", wraplength=600, justify="left").pack(anchor="w")
        
        tk.Button(content, text="CLOSE", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=30, cursor="hand2",
                relief="flat", bd=0, command=explanation_window.destroy).pack(pady=30, ipady=12)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def show_quick_borrow_by_id(self):
        """Quick borrow book by entering Book ID from QR scan"""
        input_window = tk.Toplevel(self.root)
        input_window.title("Quick Borrow - Enter Book ID")
        input_window.geometry("500x400")
        input_window.configure(bg="#1a1a2e")
        input_window.grab_set()
        
        # Header
        header = tk.Frame(input_window, bg="#27ae60")
        header.pack(fill="x")
        
        tk.Label(header, text="📚 Quick Borrow by Book ID", font=("Arial", 20, "bold"),
                bg="#27ae60", fg="white").pack(pady=20)
        
        # Instructions
        instructions = tk.Frame(input_window, bg="#0f3460")
        instructions.pack(fill="x", padx=20, pady=20)
        
        inst_content = tk.Frame(instructions, bg="#0f3460")
        inst_content.pack(padx=20, pady=15)
        
        tk.Label(inst_content, text="📱 How to use:", font=("Arial", 12, "bold"),
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 10))
        
        steps = [
            "1. Print or view the book's QR code",
            "2. Scan it with your phone's QR scanner",
            "3. Phone will show: LibraAI-Book-ID:123|Title:...",
            "4. Enter the Book ID number below (e.g., 123)"
        ]
        
        for step in steps:
            tk.Label(inst_content, text=step, font=("Arial", 10),
                    bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=2)
        
        # Input form
        form_frame = tk.Frame(input_window, bg="#1a1a2e")
        form_frame.pack(fill="x", padx=40, pady=20)
        
        tk.Label(form_frame, text="Book ID *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        book_id_entry = tk.Entry(form_frame, font=("Arial", 14), width=30,
                                bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                                relief="flat", bd=0)
        book_id_entry.pack(ipady=12)
        book_id_entry.focus()
        
        tk.Label(form_frame, text="Example: 1, 5, 23", font=("Arial", 9, "italic"),
                bg="#1a1a2e", fg="#7f8c8d").pack(anchor="w", pady=5)
        
        # Buttons
        btn_frame = tk.Frame(input_window, bg="#1a1a2e")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="✓ BORROW BOOK", font=("Arial", 12, "bold"),
                bg="#27ae60", fg="white", width=18, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.quick_borrow_by_id(book_id_entry.get(), input_window)).pack(
                    side="left", padx=10, ipady=12)
        
        tk.Button(btn_frame, text="✗ CANCEL", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=18, cursor="hand2",
                relief="flat", bd=0,
                command=input_window.destroy).pack(side="left", padx=10, ipady=12)
        
        # Bind Enter key
        book_id_entry.bind('<Return>', lambda e: self.quick_borrow_by_id(book_id_entry.get(), input_window))


    # STEP 2: ADD QUICK RETURN BY ID METHOD
    # Add this NEW method:

    def show_quick_return_by_id(self):
        """Quick return book by entering Book ID from QR scan"""
        input_window = tk.Toplevel(self.root)
        input_window.title("Quick Return - Enter Book ID")
        input_window.geometry("500x400")
        input_window.configure(bg="#1a1a2e")
        input_window.grab_set()
        
        # Header
        header = tk.Frame(input_window, bg="#3498db")
        header.pack(fill="x")
        
        tk.Label(header, text="📥 Quick Return by Book ID", font=("Arial", 20, "bold"),
                bg="#3498db", fg="white").pack(pady=20)
        
        # Instructions
        instructions = tk.Frame(input_window, bg="#0f3460")
        instructions.pack(fill="x", padx=20, pady=20)
        
        inst_content = tk.Frame(instructions, bg="#0f3460")
        inst_content.pack(padx=20, pady=15)
        
        tk.Label(inst_content, text="📱 How to use:", font=("Arial", 12, "bold"),
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 10))
        
        steps = [
            "1. Scan the book's QR code with your phone",
            "2. Phone will show the Book ID number",
            "3. Enter the Book ID below to return the book"
        ]
        
        for step in steps:
            tk.Label(inst_content, text=step, font=("Arial", 10),
                    bg="#0f3460", fg="#a0a0a0").pack(anchor="w", pady=2)
        
        # Input form
        form_frame = tk.Frame(input_window, bg="#1a1a2e")
        form_frame.pack(fill="x", padx=40, pady=20)
        
        tk.Label(form_frame, text="Book ID *", font=("Arial", 11, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        book_id_entry = tk.Entry(form_frame, font=("Arial", 14), width=30,
                                bg="#0f3460", fg="#ffffff", insertbackground="#ffffff",
                                relief="flat", bd=0)
        book_id_entry.pack(ipady=12)
        book_id_entry.focus()
        
        tk.Label(form_frame, text="Example: 1, 5, 23", font=("Arial", 9, "italic"),
                bg="#1a1a2e", fg="#7f8c8d").pack(anchor="w", pady=5)
        
        # Buttons
        btn_frame = tk.Frame(input_window, bg="#1a1a2e")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="✓ RETURN BOOK", font=("Arial", 12, "bold"),
                bg="#3498db", fg="white", width=18, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.quick_return_by_id(book_id_entry.get(), input_window)).pack(
                    side="left", padx=10, ipady=12)
        
        tk.Button(btn_frame, text="✗ CANCEL", font=("Arial", 12, "bold"),
                bg="#95a5a6", fg="white", width=18, cursor="hand2",
                relief="flat", bd=0,
                command=input_window.destroy).pack(side="left", padx=10, ipady=12)
        
        # Bind Enter key
        book_id_entry.bind('<Return>', lambda e: self.quick_return_by_id(book_id_entry.get(), input_window))


    # STEP 3: ADD BORROW PROCESSOR
    # Add this NEW method:

    def quick_borrow_by_id(self, book_id_str, window):
        """Process quick borrow by Book ID"""
        from datetime import datetime, timedelta
        
        # Validate input
        if not book_id_str or not book_id_str.strip():
            messagebox.showwarning("Input Error", "Please enter a Book ID")
            return
        
        try:
            book_id = int(book_id_str.strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Book ID must be a number")
            return
        
        # Get book details
        self.cursor.execute("""
            SELECT title, author, available FROM books WHERE book_id = %s
        """, (book_id,))
        
        result = self.cursor.fetchone()
        
        if not result:
            messagebox.showerror("Not Found", f"Book ID {book_id} not found in database!")
            return
        
        title, author, available = result
        
        # Check availability
        if available <= 0:
            response = messagebox.askyesno("Not Available", 
                                        f"'{title}' is currently not available.\n\nWould you like to reserve it?")
            if response:
                window.destroy()
                self.reserve_book(book_id)
            return
        
        # Check if already borrowed
        self.cursor.execute("""
            SELECT * FROM transactions 
            WHERE user_id = %s AND book_id = %s AND status = 'borrowed'
        """, (self.current_user['user_id'], book_id))
        
        if self.cursor.fetchone():
            messagebox.showerror("Already Borrowed", f"You have already borrowed '{title}'")
            return
        
        # Create transaction
        borrow_date = datetime.now().date()
        due_date = borrow_date + timedelta(days=14)
        
        try:
            self.cursor.execute("""
                INSERT INTO transactions (user_id, book_id, borrow_date, due_date, status)
                VALUES (%s, %s, %s, %s, 'borrowed')
            """, (self.current_user['user_id'], book_id, borrow_date, due_date))
            
            # Update availability
            self.cursor.execute("""
                UPDATE books SET available = available - 1 WHERE book_id = %s
            """, (book_id,))
            
            self.db.commit()
            
            window.destroy()
            
            messagebox.showinfo("Success! 📚", 
                            f"Book borrowed successfully!\n\n"
                            f"📖 {title}\n"
                            f"✍️ by {author}\n\n"
                            f"📅 Due Date: {due_date}\n\n"
                            f"Please return on time to avoid penalties.")
            
            # Refresh view
            if hasattr(self, 'results_container'):
                self.show_search_books()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to borrow book: {str(e)}")


    # STEP 4: ADD RETURN PROCESSOR
    # Add this NEW method:

    def quick_return_by_id(self, book_id_str, window):
        """Process quick return by Book ID"""
        from datetime import datetime
        
        # Validate input
        if not book_id_str or not book_id_str.strip():
            messagebox.showwarning("Input Error", "Please enter a Book ID")
            return
        
        try:
            book_id = int(book_id_str.strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Book ID must be a number")
            return
        
        # Get book details
        self.cursor.execute("""
            SELECT title, author FROM books WHERE book_id = %s
        """, (book_id,))
        
        result = self.cursor.fetchone()
        
        if not result:
            messagebox.showerror("Not Found", f"Book ID {book_id} not found in database!")
            return
        
        title, author = result
        
        # Check if user has borrowed this book
        self.cursor.execute("""
            SELECT transaction_id, due_date FROM transactions 
            WHERE user_id = %s AND book_id = %s AND status = 'borrowed'
        """, (self.current_user['user_id'], book_id))
        
        trans_result = self.cursor.fetchone()
        
        if not trans_result:
            messagebox.showerror("Not Borrowed", 
                            f"You haven't borrowed '{title}' or it's already returned.")
            return
        
        trans_id, due_date = trans_result
        return_date = datetime.now().date()
        
        # Calculate penalty if overdue
        penalty_message = ""
        if return_date > due_date:
            days_overdue = (return_date - due_date).days
            penalty_amount = days_overdue * 1.00
            
            self.cursor.execute("""
                INSERT INTO penalties (transaction_id, user_id, amount, days_overdue)
                VALUES (%s, %s, %s, %s)
            """, (trans_id, self.current_user['user_id'], penalty_amount, days_overdue))
            
            penalty_message = f"\n\n⚠️ OVERDUE NOTICE\nDays overdue: {days_overdue}\nPenalty: RM {penalty_amount:.2f}\n\nPlease settle at the library counter."
        
        # Update transaction
        self.cursor.execute("""
            UPDATE transactions 
            SET return_date = %s, status = 'returned'
            WHERE transaction_id = %s
        """, (return_date, trans_id))
        
        # Update availability
        self.cursor.execute("""
            UPDATE books SET available = available + 1 WHERE book_id = %s
        """, (book_id,))
        
        self.db.commit()
        
        # Check for reservations
        self.check_available_reservations()
        
        window.destroy()
        
        success_message = f"Book returned successfully! ✅\n\n📖 {title}\n✍️ by {author}{penalty_message}"
        
        if penalty_message:
            messagebox.showwarning("Book Returned", success_message)
        else:
            messagebox.showinfo("Success!", success_message)
        
        # Refresh view
        if hasattr(self, 'borrowed_tree'):
            self.show_my_books()


    def send_due_date_reminder_email(self, user_email, user_name, book_title, due_date):
        """Send due date reminder email"""
        subject = "📅 Book Due Soon - Return Reminder"
        
        content = f"""
        <p>Dear <strong>{user_name}</strong>,</p>
        
        <div class="warning">
            <p><strong>⏰ Reminder: Book Due Soon</strong></p>
            <p>Your borrowed book is due in 2 days. Please return it on time to avoid penalties.</p>
        </div>
        
        <p><strong>Book Details:</strong></p>
        <ul>
            <li>📖 Title: {book_title}</li>
            <li>📅 Due Date: {due_date}</li>
            <li>💰 Penalty: RM 1.00 per day if overdue</li>
        </ul>
        
        <p>Please return the book to the library before the due date.</p>
        
        <p>Thank you for using LibraAI!</p>
        """
        
        html = self.email_service.create_html_template(
            "Book Due Date Reminder", content
        )
        
        return self.email_service.send_email(user_email, subject, html)


    def send_overdue_notification_email(self, user_email, user_name, book_title, days_overdue, penalty_amount):
        """Send overdue book notification email"""
        subject = "⚠️ URGENT: Overdue Book - Penalty Applied"
        
        content = f"""
        <p>Dear <strong>{user_name}</strong>,</p>
        
        <div class="warning">
            <p><strong>⚠️ OVERDUE NOTICE</strong></p>
            <p>Your borrowed book is now <strong>{days_overdue} day(s) overdue</strong>.</p>
        </div>
        
        <p><strong>Book Details:</strong></p>
        <ul>
            <li>📖 Title: {book_title}</li>
            <li>⏱️ Days Overdue: {days_overdue} days</li>
            <li>💰 Current Penalty: RM {penalty_amount:.2f}</li>
        </ul>
        
        <p><strong>Action Required:</strong></p>
        <p>Please return the book immediately to avoid additional penalties. 
        Penalties increase by RM 1.00 per day.</p>
        
        <p>Please visit the library counter to return the book and settle your penalty.</p>
        """
        
        html = self.email_service.create_html_template(
            "Overdue Book Notification", content
        )
        
        return self.email_service.send_email(user_email, subject, html)


    def send_penalty_notice_email(self, user_email, user_name, book_title, penalty_amount):
        """Send penalty notice email"""
        subject = "💰 Penalty Notice - Payment Required"
        
        content = f"""
        <p>Dear <strong>{user_name}</strong>,</p>
        
        <p>A penalty has been applied to your account for returning a book late.</p>
        
        <p><strong>Penalty Details:</strong></p>
        <ul>
            <li>📖 Book: {book_title}</li>
            <li>💰 Amount: RM {penalty_amount:.2f}</li>
            <li>📍 Payment: Library counter</li>
        </ul>
        
        <p>Please settle this penalty at your earliest convenience.</p>
        
        <p>Thank you for your cooperation.</p>
        """
        
        html = self.email_service.create_html_template(
            "Penalty Notice", content
        )
        
        return self.email_service.send_email(user_email, subject, html)


    def send_reservation_available_email(self, user_email, user_name, book_title, expires_at):
        """Send email when reserved book becomes available"""
        subject = "✅ Reserved Book Now Available!"
        
        content = f"""
        <p>Dear <strong>{user_name}</strong>,</p>
        
        <div class="info">
            <p><strong>🎉 Good News!</strong></p>
            <p>Your reserved book is now available for borrowing.</p>
        </div>
        
        <p><strong>Book Details:</strong></p>
        <ul>
            <li>📖 Title: {book_title}</li>
            <li>⏰ Available Until: {expires_at}</li>
            <li>⚠️ Claim within 2 days or reservation expires</li>
        </ul>
        
        <a href="#" class="button">Visit Library to Borrow</a>
        
        <p>Please visit the library to borrow this book within 2 days.</p>
        """
        
        html = self.email_service.create_html_template(
            "Reserved Book Available", content
        )
        
        return self.email_service.send_email(user_email, subject, html)


    def send_welcome_email(self, user_email, user_name, username):
        """Send welcome email to new users"""
        subject = "🎉 Welcome to LibraAI Library System!"
        
        content = f"""
        <p>Dear <strong>{user_name}</strong>,</p>
        
        <p>Welcome to LibraAI - Your AI-Powered Library Management System!</p>
        
        <div class="info">
            <p><strong>Your Account Details:</strong></p>
            <ul>
                <li>👤 Username: {username}</li>
                <li>📧 Email: {user_email}</li>
            </ul>
        </div>
        
        <p><strong>Getting Started:</strong></p>
        <ol>
            <li>🔍 Search and browse our book collection</li>
            <li>📚 Borrow books for 14 days</li>
            <li>🤖 Get AI-powered recommendations</li>
            <li>📋 Reserve unavailable books</li>
            <li>🔔 Receive notifications and reminders</li>
        </ol>
        
        <a href="#" class="button">Start Exploring</a>
        
        <p><strong>Important:</strong></p>
        <ul>
            <li>Return books on time to avoid penalties (RM 1.00/day)</li>
            <li>Check your notifications regularly</li>
            <li>Update your profile information</li>
        </ul>
        
        <p>Happy reading! 📖</p>
        """
        
        html = self.email_service.create_html_template(
            "Welcome to LibraAI", content
        )
        
        return self.email_service.send_email(user_email, subject, html)
    def send_password_reset_email(self, recipient_email, user_name, reset_code):
        """Send password reset code email"""
        subject = "🔐 Password Reset Code - LibraAI"
        
        content = f"""
        <p>Dear <strong>{user_name}</strong>,</p>
        
        <p>We received a request to reset your password for your LibraAI account.</p>
        
        <div class="warning">
            <p><strong>🔑 Your Password Reset Code:</strong></p>
            <h1 style="text-align: center; font-size: 48px; color: #e94560; letter-spacing: 5px;">
                {reset_code}
            </h1>
        </div>
        
        <p><strong>Important:</strong></p>
        <ul>
            <li>⏰ This code expires in 15 minutes</li>
            <li>🔒 Do not share this code with anyone</li>
            <li>❌ If you didn't request this, please ignore this email</li>
        </ul>
        
        <p>Enter this code in the password reset form to create a new password.</p>
        
        <p>If you have any concerns, please contact the library administrator.</p>
        """
        
        html = self.create_html_template(
            "Password Reset Request", content
        )
        
        return self.send_email(recipient_email, subject, html)

    def show_forgot_password(self):
        """Display forgot password screen"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True)
        
        # Center container
        center_frame = tk.Frame(main_frame, bg="#0f3460", padx=60, pady=40)
        center_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Header
        tk.Label(center_frame, text="🔐 Forgot Password", font=("Arial", 28, "bold"),
                bg="#0f3460", fg="#ffffff").pack(pady=(0, 10))
        tk.Label(center_frame, text="Enter your email to receive a reset code", 
                font=("Arial", 12), bg="#0f3460", fg="#a0a0a0").pack(pady=(0, 30))
        
        # Email field
        tk.Label(center_frame, text="Email Address", font=("Arial", 11, "bold"), 
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        email_entry = tk.Entry(center_frame, font=("Arial", 12), width=40,
                            bg="#1a1a2e", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        email_entry.pack(ipady=10, pady=(0, 30))
        
        # Buttons
        btn_frame = tk.Frame(center_frame, bg="#0f3460")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="SEND RESET CODE", font=("Arial", 11, "bold"),
                bg="#3498db", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.send_reset_code(email_entry.get())).pack(
                    side="left", padx=10, ipady=12)
        
        tk.Button(btn_frame, text="BACK TO LOGIN", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=self.show_login).pack(side="left", padx=10, ipady=12)
        
        email_entry.bind('<Return>', lambda e: self.send_reset_code(email_entry.get()))


    # STEP 5: ADD SEND RESET CODE METHOD
    # Add this NEW method:

    def send_reset_code(self, email):
        """Generate and send password reset code"""
        import random
        from datetime import datetime, timedelta
        
        email = email.strip()
        
        # Validate email
        if not email or not self.validate_email(email):
            messagebox.showerror("Invalid Email", "Please enter a valid email address")
            return
        
        # Check if user exists
        self.cursor.execute("""
            SELECT user_id, full_name, username FROM users WHERE email = %s
        """, (email,))
        
        result = self.cursor.fetchone()
        
        if not result:
            messagebox.showerror("Not Found", "No account found with this email address")
            return
        
        user_id, full_name, username = result
        
        # Generate 6-digit code
        reset_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Set expiry (15 minutes)
        expires_at = datetime.now() + timedelta(minutes=15)
        
        try:
            # Store reset code
            self.cursor.execute("""
                INSERT INTO password_resets (user_id, reset_code, expires_at)
                VALUES (%s, %s, %s)
            """, (user_id, reset_code, expires_at))
            
            self.db.commit()
            
            # Send email
            success, message = self.email_service.send_password_reset_email(
                email, full_name, reset_code
            )
            
            if success:
                messagebox.showinfo("Email Sent! 📧", 
                                f"A 6-digit reset code has been sent to:\n{email}\n\n"
                                f"The code will expire in 15 minutes.\n\n"
                                f"Please check your email and enter the code.")
                
                # Show reset code entry screen
                self.show_reset_code_entry(email)
            else:
                messagebox.showerror("Email Failed", 
                                f"Failed to send email: {message}\n\n"
                                f"Please check your email configuration.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate reset code: {str(e)}")


    # STEP 6: ADD RESET CODE ENTRY SCREEN
    # Add this NEW method:

    def show_reset_code_entry(self, email):
        """Display reset code entry and new password form"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True)
        
        # Scrollable frame
        canvas = tk.Canvas(main_frame, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Center container
        center_frame = tk.Frame(scrollable_frame, bg="#0f3460", padx=60, pady=40)
        center_frame.pack(pady=50)
        
        # Header
        tk.Label(center_frame, text="🔑 Enter Reset Code", font=("Arial", 28, "bold"),
                bg="#0f3460", fg="#ffffff").pack(pady=(0, 10))
        tk.Label(center_frame, text=f"Code sent to: {email}", 
                font=("Arial", 11), bg="#0f3460", fg="#27ae60").pack(pady=(0, 30))
        
        # Reset Code
        tk.Label(center_frame, text="Reset Code (6 digits) *", font=("Arial", 11, "bold"), 
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        code_entry = tk.Entry(center_frame, font=("Arial", 16), width=35,
                            bg="#1a1a2e", fg="#ffffff", insertbackground="#ffffff",
                            relief="flat", bd=0)
        code_entry.pack(ipady=12, pady=(0, 20))
        code_entry.focus()
        
        # New Password
        tk.Label(center_frame, text="New Password *", font=("Arial", 11, "bold"), 
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(0, 5))
        
        new_pwd_entry = tk.Entry(center_frame, font=("Arial", 12), width=40, show="●",
                                bg="#1a1a2e", fg="#ffffff", insertbackground="#ffffff",
                                relief="flat", bd=0)
        new_pwd_entry.pack(ipady=10, pady=(0, 10))
        
        # Password strength indicator
        strength_label = tk.Label(center_frame, text="", font=("Arial", 9),
                                bg="#0f3460")
        strength_label.pack(pady=5)
        
        new_pwd_entry.bind('<KeyRelease>', lambda e: self.check_password_strength(
            new_pwd_entry.get(), strength_label))
        
        # Confirm Password
        tk.Label(center_frame, text="Confirm Password *", font=("Arial", 11, "bold"), 
                bg="#0f3460", fg="#ffffff").pack(anchor="w", pady=(10, 5))
        
        confirm_pwd_entry = tk.Entry(center_frame, font=("Arial", 12), width=40, show="●",
                                    bg="#1a1a2e", fg="#ffffff", insertbackground="#ffffff",
                                    relief="flat", bd=0)
        confirm_pwd_entry.pack(ipady=10, pady=(0, 20))
        
        # Password requirements
        req_frame = tk.Frame(center_frame, bg="#1a1a2e", relief="flat", bd=0)
        req_frame.pack(fill="x", pady=10)
        
        tk.Label(req_frame, text="Password Requirements:", font=("Arial", 10, "bold"),
                bg="#1a1a2e", fg="#ffffff").pack(anchor="w", padx=15, pady=(10, 5))
        
        reqs = [
            "• At least 8 characters",
            "• One uppercase letter (A-Z)",
            "• One lowercase letter (a-z)",
            "• One number (0-9)",
            "• One special character (@$!%*?&#)"
        ]
        
        for req in reqs:
            tk.Label(req_frame, text=req, font=("Arial", 9),
                    bg="#1a1a2e", fg="#a0a0a0").pack(anchor="w", padx=30, pady=2)
        
        # Buttons
        btn_frame = tk.Frame(center_frame, bg="#0f3460")
        btn_frame.pack(pady=30)
        
        tk.Button(btn_frame, text="RESET PASSWORD", font=("Arial", 11, "bold"),
                bg="#27ae60", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=lambda: self.verify_and_reset_password(
                    email, code_entry.get(), new_pwd_entry.get(), 
                    confirm_pwd_entry.get())).pack(side="left", padx=10, ipady=12)
        
        tk.Button(btn_frame, text="BACK TO LOGIN", font=("Arial", 11, "bold"),
                bg="#95a5a6", fg="white", width=20, cursor="hand2",
                relief="flat", bd=0,
                command=self.show_login).pack(side="left", padx=10, ipady=12)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


    # STEP 7: ADD PASSWORD RESET VERIFICATION
    # Add this NEW method:

    def verify_and_reset_password(self, email, code, new_password, confirm_password):
        """Verify reset code and update password"""
        from datetime import datetime
        
        # Validate inputs
        if not code or not new_password or not confirm_password:
            messagebox.showwarning("Input Error", "Please fill in all fields")
            return
        
        if len(code) != 6 or not code.isdigit():
            messagebox.showerror("Invalid Code", "Reset code must be 6 digits")
            return
        
        # Validate password
        is_valid, message = self.validate_password(new_password)
        if not is_valid:
            messagebox.showerror("Weak Password", message)
            return
        
        # Check passwords match
        if new_password != confirm_password:
            messagebox.showerror("Password Mismatch", "Passwords do not match")
            return
        
        # Get user
        self.cursor.execute("""
            SELECT user_id FROM users WHERE email = %s
        """, (email,))
        
        user_result = self.cursor.fetchone()
        if not user_result:
            messagebox.showerror("Error", "User not found")
            return
        
        user_id = user_result[0]
        
        # Verify reset code
        self.cursor.execute("""
            SELECT reset_id, expires_at FROM password_resets
            WHERE user_id = %s AND reset_code = %s AND used = FALSE
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id, code))
        
        reset_result = self.cursor.fetchone()
        
        if not reset_result:
            messagebox.showerror("Invalid Code", 
                            "Reset code is invalid or has already been used.\n\n"
                            "Please request a new code.")
            return
        
        reset_id, expires_at = reset_result
        
        # Check if expired
        if datetime.now() > expires_at:
            messagebox.showerror("Code Expired", 
                            "This reset code has expired (15 minutes limit).\n\n"
                            "Please request a new code.")
            return
        
        # Update password
        try:
            hashed_pwd = self.hash_password(new_password)
            
            self.cursor.execute("""
                UPDATE users SET password = %s WHERE user_id = %s
            """, (hashed_pwd, user_id))
            
            # Mark reset code as used
            self.cursor.execute("""
                UPDATE password_resets SET used = TRUE WHERE reset_id = %s
            """, (reset_id,))
            
            self.db.commit()
            
            messagebox.showinfo("Success! ✅", 
                            "Your password has been reset successfully!\n\n"
                            "You can now login with your new password.")
            
            self.show_login()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset password: {str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = LibraAISystem(root)
    root.mainloop()