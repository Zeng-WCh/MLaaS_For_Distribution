from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class UserDBLine(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(256), unique=True, nullable=False)
    password = db.Column(db.String(256))
    email = db.Column(db.String(256), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())


class TrainDBLine(db.Model):
    __tablename__ = 'training_records'
    id = db.Column(db.Integer, primary_key=True)
    submit_date = db.Column(db.DateTime, nullable=False)
    model_choice = db.Column(db.String(256), nullable=False)
    # is_completed = db.Column(db.Boolean)
    # 枚举类型
    training_status = db.Column(db.Enum('pending', 'running',
                                        'completed', 'failed'), default='pending')
    created_by = db.Column(db.String(256), db.ForeignKey(
        'user.username'), nullable=False)
    completed_at = db.Column(db.DateTime)
    model_location = db.Column(db.String(256))
    model_name = db.Column(db.String(256))
