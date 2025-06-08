import json
import re
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def download_nltk_resources():
    """Загрузка всех необходимых ресурсов NLTK"""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Загружаем ресурс NLTK: {resource_name}")
            nltk.download(resource_name, quiet=True)


download_nltk_resources()


@dataclass
class TestCase:
    """Структура для хранения тест-кейса"""
    name: str
    id: str
    preconditions: str
    steps: List[str]
    expected_result: str
    raw_text: str


@dataclass
class QualityMetrics:
    """Структура для хранения метрик качества"""
    completeness: float
    semantic_relevance: float
    diversity: float
    language_quality: float
    redundancy: float
    overall_score: float


class TestCaseParser:
    """Парсер для извлечения тест-кейсов из текста"""

    def parse_test_cases(self, text: str) -> List[TestCase]:
        """Парсинг тест-кейсов из текста"""

        test_cases = []

        # Разделение на отдельные тест-кейсы
        sections = re.split(r'\n\s*(?=Название:|ID:|Test Case|Тест-кейс)', text, flags=re.IGNORECASE)

        for section in sections:
            if not section.strip():
                continue

            test_case = self._parse_single_test_case(section)
            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _parse_single_test_case(self, text: str) -> TestCase:
        """Парсинг одного тест-кейса"""

        try:
            # Извлечение названия
            name_match = re.search(r'Название:\s*(.*?)(?=\n|ID:|$)', text, re.IGNORECASE | re.DOTALL)
            name = name_match.group(1).strip() if name_match else "Не указано"

            # Извлечение ID
            id_match = re.search(r'ID:\s*(.*?)(?=\n|Предусловия:|$)', text, re.IGNORECASE)
            test_id = id_match.group(1).strip() if id_match else "Не указано"

            # Извлечение предусловий
            precond_match = re.search(r'Предусловия:\s*(.*?)(?=\nШаги:|$)', text, re.IGNORECASE | re.DOTALL)
            preconditions = precond_match.group(1).strip() if precond_match else "Не указано"

            # Извлечение шагов
            steps_match = re.search(r'Шаги:\s*(.*?)(?=\nОжидаемый результат:|$)', text, re.IGNORECASE | re.DOTALL)
            steps_text = steps_match.group(1) if steps_match else ""
            steps = self._extract_steps(steps_text)

            # Извлечение ожидаемого результата
            result_match = re.search(r'Ожидаемый результат:\s*(.*?)(?=\n\n|$)', text, re.IGNORECASE | re.DOTALL)
            expected_result = result_match.group(1).strip() if result_match else "Не указано"

            return TestCase(
                name=name,
                id=test_id,
                preconditions=preconditions,
                steps=steps,
                expected_result=expected_result,
                raw_text=text
            )
        except Exception as e:
            print(f"Ошибка при парсинге тест-кейса: {e}")
            return None

    def _extract_steps(self, steps_text: str) -> List[str]:
        """Извлечение списка шагов"""

        steps = []
        lines = steps_text.split('\n')

        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                steps.append(line)

        return steps


class QualityAnalyzer:
    """Анализатор качества тест-кейсов"""

    def __init__(self):
        try:
            self.russian_stopwords = set(stopwords.words('russian'))
        except Exception:
            print("Предупреждение: русские стоп-слова недоступны")
            self.russian_stopwords = set()

        try:
            self.english_stopwords = set(stopwords.words('english'))
        except Exception:
            print("Предупреждение: английские стоп-слова недоступны")
            self.english_stopwords = set()

        self.all_stopwords = self.russian_stopwords.union(self.english_stopwords)

        self.function_keywords = {
            'dijkstra': ['граф', 'путь', 'вершина', 'ребро', 'вес', 'кратчайший', 'алгоритм', 'дейкстра'],
            'ui': ['кнопка', 'интерфейс', 'клик', 'поле', 'форма', 'элемент', 'страница', 'ui'],
            'atbash': ['шифр', 'атбаш', 'декодирование', 'кодирование', 'символ', 'буква', 'текст'],
            'harris_benedict': ['калькулятор', 'метаболизм', 'калории', 'вес', 'рост', 'возраст', 'пол', 'харрис']
        }

    def analyze_test_cases(self, test_cases: List[TestCase], function_type: str = None) -> QualityMetrics:
        """Комплексный анализ качества тест-кейсов"""

        if not test_cases:
            return QualityMetrics(0, 0, 0, 0, 1, 0)

        completeness = self._calculate_completeness(test_cases, function_type)
        semantic_relevance = self._calculate_semantic_relevance(test_cases, function_type)
        diversity = self._calculate_diversity(test_cases)
        language_quality = self._calculate_language_quality(test_cases)
        redundancy = self._calculate_redundancy(test_cases)

        # Общий балл
        overall_score = (
                completeness * 0.25 +
                semantic_relevance * 0.25 +
                diversity * 0.2 +
                language_quality * 0.15 +
                (1 - redundancy) * 0.15  # Инвертируем redundancy
        )

        return QualityMetrics(
            completeness=completeness,
            semantic_relevance=semantic_relevance,
            diversity=diversity,
            language_quality=language_quality,
            redundancy=redundancy,
            overall_score=overall_score
        )

    def _calculate_completeness(self, test_cases: List[TestCase], function_type: str) -> float:
        """Расчет полноты покрытия"""

        if not test_cases:
            return 0.0

        coverage_score = 0.0
        total_checks = 6

        positive_cases = sum(1 for tc in test_cases if self._is_positive_case(tc))
        coverage_score += min(positive_cases / max(len(test_cases) * 0.6, 1), 1.0)

        negative_cases = sum(1 for tc in test_cases if self._is_negative_case(tc))
        coverage_score += min(negative_cases / max(len(test_cases) * 0.3, 1), 1.0)

        boundary_cases = sum(1 for tc in test_cases if self._is_boundary_case(tc))
        coverage_score += min(boundary_cases / max(len(test_cases) * 0.2, 1), 1.0)

        case_count_score = min(len(test_cases) / 8, 1.0)
        coverage_score += case_count_score

        input_diversity = self._calculate_input_diversity(test_cases)
        coverage_score += input_diversity

        if function_type:
            type_specific_score = self._calculate_type_specific_coverage(test_cases, function_type)
            coverage_score += type_specific_score
        else:
            coverage_score += 0.5

        return coverage_score / total_checks

    def _is_positive_case(self, test_case: TestCase) -> bool:
        """Определение позитивного тест-кейса"""
        positive_indicators = ['успешно', 'корректно', 'правильно', 'валидн', 'норм']
        text = (test_case.name + ' ' + test_case.expected_result).lower()
        return any(indicator in text for indicator in positive_indicators)

    def _is_negative_case(self, test_case: TestCase) -> bool:
        """Определение негативного тест-кейса"""
        negative_indicators = ['ошибка', 'неверн', 'невалидн', 'исключение', 'отрицател', 'пуст']
        text = (test_case.name + ' ' + test_case.expected_result).lower()
        return any(indicator in text for indicator in negative_indicators)

    def _is_boundary_case(self, test_case: TestCase) -> bool:
        """Определение граничного случая"""
        boundary_indicators = ['граничн', 'предельн', 'максимальн', 'минимальн', 'край', 'ноль', 'бесконечност']
        text = (test_case.name + ' ' + test_case.preconditions + ' ' + test_case.expected_result).lower()
        return any(indicator in text for indicator in boundary_indicators)

    def _calculate_input_diversity(self, test_cases: List[TestCase]) -> float:
        """Расчет разнообразия входных данных"""
        if not test_cases:
            return 0.0

        input_texts = [tc.preconditions.lower() for tc in test_cases if tc.preconditions != "Не указано"]
        if not input_texts:
            return 0.0

        # TF-IDF для оценки
        try:
            vectorizer = TfidfVectorizer(stop_words=list(self.all_stopwords), max_features=100)
            tfidf_matrix = vectorizer.fit_transform(input_texts)

            # Средняя попарная схожесть
            similarities = cosine_similarity(tfidf_matrix)
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            return 1 - avg_similarity
        except:
            return 0.5

    def _calculate_type_specific_coverage(self, test_cases: List[TestCase], function_type: str) -> float:
        """Расчет специфического покрытия для типа функции"""
        if function_type not in self.function_keywords:
            return 0.5

        keywords = self.function_keywords[function_type]
        total_score = 0.0

        for test_case in test_cases:
            case_text = (test_case.name + ' ' + test_case.preconditions + ' ' +
                         ' '.join(test_case.steps) + ' ' + test_case.expected_result).lower()

            keyword_matches = sum(1 for keyword in keywords if keyword in case_text)
            case_score = min(keyword_matches / len(keywords), 1.0)
            total_score += case_score

        return total_score / len(test_cases) if test_cases else 0.0

    def _calculate_semantic_relevance(self, test_cases: List[TestCase], function_type: str) -> float:
        """Расчет семантической релевантности"""
        if not test_cases:
            return 0.0

        total_relevance = 0.0

        for test_case in test_cases:
            case_relevance = 0.0

            steps_coherence = self._calculate_steps_coherence(test_case)
            case_relevance += steps_coherence * 0.4

            precondition_coherence = self._calculate_precondition_coherence(test_case)
            case_relevance += precondition_coherence * 0.3

            realism_score = self._calculate_realism(test_case)
            case_relevance += realism_score * 0.3

            total_relevance += case_relevance

        return total_relevance / len(test_cases)

    def _calculate_steps_coherence(self, test_case: TestCase) -> float:
        """Расчет связности шагов"""
        if not test_case.steps or test_case.expected_result == "Не указано":
            return 0.0

        steps_text = ' '.join(test_case.steps).lower()
        result_text = test_case.expected_result.lower()

        coherence_indicators = ['затем', 'после', 'далее', 'следующий', 'результат', 'получ']
        coherence_score = sum(1 for indicator in coherence_indicators if indicator in steps_text)

        numbered_steps = len([s for s in test_case.steps if re.match(r'^\d+\.', s.strip())])
        sequence_score = min(numbered_steps / len(test_case.steps), 1.0) if test_case.steps else 0

        return min((coherence_score * 0.1 + sequence_score) / 1.1, 1.0)

    def _calculate_precondition_coherence(self, test_case: TestCase) -> float:
        """Расчет связности предусловий и шагов"""
        if test_case.preconditions == "Не указано" or not test_case.steps:
            return 0.5

        try:
            precond_words = set(word_tokenize(test_case.preconditions.lower()))
            steps_words = set(word_tokenize(' '.join(test_case.steps).lower()))

            precond_words -= self.all_stopwords
            steps_words -= self.all_stopwords

            if not precond_words or not steps_words:
                return 0.5

            intersection = len(precond_words.intersection(steps_words))
            union = len(precond_words.union(steps_words))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            print(f"Ошибка при анализе связности предусловий: {e}")
            return 0.5

    def _calculate_realism(self, test_case: TestCase) -> float:
        """Расчет реалистичности тест-кейса"""
        realism_score = 1.0

        generic_phrases = ['проверить', 'убедиться', 'выполнить', 'сделать']
        case_text = (test_case.name + ' ' + ' '.join(test_case.steps)).lower()

        generic_count = sum(1 for phrase in generic_phrases if phrase in case_text)
        if generic_count > len(test_case.steps) * 0.5:
            realism_score *= 0.7

        # Проверка на наличие конкретных данных
        has_specific_data = bool(re.search(r'\d+|[а-яё]+@[а-яё]+\.[а-яё]+|"[^"]*"', case_text))
        if has_specific_data:
            realism_score *= 1.2

        return min(realism_score, 1.0)

    def _calculate_diversity(self, test_cases: List[TestCase]) -> float:
        """Расчет разнообразия тест-кейсов"""
        if len(test_cases) < 2:
            return 0.0

        case_texts = []
        for tc in test_cases:
            case_text = (tc.name + ' ' + tc.preconditions + ' ' +
                         ' '.join(tc.steps) + ' ' + tc.expected_result)
            case_texts.append(case_text.lower())

        try:
            vectorizer = TfidfVectorizer(
                stop_words=list(self.all_stopwords),
                max_features=200,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(case_texts)

            similarities = cosine_similarity(tfidf_matrix)

            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            avg_similarity = np.mean(upper_triangle)

            diversity_score = 1 - avg_similarity

            type_bonus = self._calculate_type_diversity_bonus(test_cases)

            return min(diversity_score + type_bonus, 1.0)

        except Exception:
            return 0.5

    def _calculate_type_diversity_bonus(self, test_cases: List[TestCase]) -> float:
        """Бонус за разнообразие типов тест-кейсов"""
        positive_count = sum(1 for tc in test_cases if self._is_positive_case(tc))
        negative_count = sum(1 for tc in test_cases if self._is_negative_case(tc))
        boundary_count = sum(1 for tc in test_cases if self._is_boundary_case(tc))

        total_types = (positive_count > 0) + (negative_count > 0) + (boundary_count > 0)
        return total_types / 3 * 0.1

    def _calculate_language_quality(self, test_cases: List[TestCase]) -> float:
        """Расчет качества языка"""
        if not test_cases:
            return 0.0

        total_quality = 0.0

        for test_case in test_cases:
            case_quality = 0.0

            structure_score = self._calculate_structure_score(test_case)
            case_quality += structure_score * 0.3

            clarity_score = self._calculate_clarity_score(test_case)
            case_quality += clarity_score * 0.3

            grammar_score = self._calculate_grammar_score(test_case)
            case_quality += grammar_score * 0.2

            style_score = self._calculate_style_consistency(test_case)
            case_quality += style_score * 0.2

            total_quality += case_quality

        return total_quality / len(test_cases)

    def _calculate_structure_score(self, test_case: TestCase) -> float:
        """Оценка структурированности тест-кейса"""
        score = 0.0
        total_fields = 5

        if test_case.name and test_case.name != "Не указано":
            score += 1
        if test_case.id and test_case.id != "Не указано":
            score += 1
        if test_case.preconditions and test_case.preconditions != "Не указано":
            score += 1
        if test_case.steps:
            score += 1
        if test_case.expected_result and test_case.expected_result != "Не указано":
            score += 1

        return score / total_fields

    def _calculate_clarity_score(self, test_case: TestCase) -> float:
        """Оценка четкости формулировок"""
        clarity_score = 1.0

        avg_length = np.mean([
            len(test_case.name),
            len(test_case.preconditions) if test_case.preconditions != "Не указано" else 10,
            np.mean([len(step) for step in test_case.steps]) if test_case.steps else 10,
            len(test_case.expected_result) if test_case.expected_result != "Не указано" else 10
        ])

        if avg_length < 10:
            clarity_score *= 0.6
        elif avg_length > 100:
            clarity_score *= 0.8

        vague_words = ['что-то', 'как-то', 'возможно', 'может быть', 'вероятно']
        case_text = (test_case.name + ' ' + test_case.expected_result).lower()
        vague_count = sum(1 for word in vague_words if word in case_text)

        if vague_count > 0:
            clarity_score *= 0.8

        return clarity_score

    def _calculate_grammar_score(self, test_case: TestCase) -> float:
        """Упрощенная оценка грамматической корректности"""
        grammar_score = 1.0

        case_text = test_case.raw_text

        try:
            sentences = sent_tokenize(case_text)
            sentences_with_period = sum(1 for s in sentences if s.strip().endswith('.'))

            if sentences and sentences_with_period / len(sentences) < 0.3:
                grammar_score *= 0.8

            words = word_tokenize(case_text.lower())
            word_freq = Counter(words)
            repeated_words = sum(1 for count in word_freq.values() if count > 3)

            if repeated_words > len(set(words)) * 0.1:
                grammar_score *= 0.9

        except Exception as e:
            print(f"Ошибка при анализе грамматики: {e}")
            grammar_score = 0.8

        return grammar_score

    def _calculate_style_consistency(self, test_case: TestCase) -> float:
        """Оценка единообразия стиля"""
        if not test_case.steps:
            return 0.5

        numbered_pattern = all(re.match(r'^\d+\.', step.strip()) for step in test_case.steps)
        if numbered_pattern:
            return 1.0

        step_beginnings = [step.strip()[:10].lower() for step in test_case.steps if step.strip()]
        if len(set(step_beginnings)) == len(step_beginnings):  # Все разные
            return 0.8

        return 0.6

    def _calculate_redundancy(self, test_cases: List[TestCase]) -> float:
        """Расчет избыточности тест-кейсов"""
        if len(test_cases) < 2:
            return 0.0

        redundancy_scores = []

        for i in range(len(test_cases)):
            for j in range(i + 1, len(test_cases)):
                similarity = self._calculate_test_case_similarity(test_cases[i], test_cases[j])
                redundancy_scores.append(similarity)

        avg_redundancy = np.mean(redundancy_scores)

        ids = [tc.id for tc in test_cases if tc.id != "Не указано"]
        id_duplicates = len(ids) - len(set(ids))
        id_redundancy = id_duplicates / len(test_cases) if test_cases else 0

        return (avg_redundancy + id_redundancy) / 2

    def _calculate_test_case_similarity(self, tc1: TestCase, tc2: TestCase) -> float:
        """Расчет схожести двух тест-кейсов"""
        text1 = (tc1.name + ' ' + tc1.preconditions + ' ' +
                 ' '.join(tc1.steps) + ' ' + tc1.expected_result).lower()
        text2 = (tc2.name + ' ' + tc2.preconditions + ' ' +
                 ' '.join(tc2.steps) + ' ' + tc2.expected_result).lower()

        try:
            vectorizer = TfidfVectorizer(stop_words=list(self.all_stopwords))
            tfidf_matrix = vectorizer.fit_transform([text1, text2])

            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity

        except Exception as e:
            print(f"Ошибка при расчете схожести тест-кейсов: {e}")
            try:
                words1 = set(word_tokenize(text1)) - self.all_stopwords
                words2 = set(word_tokenize(text2)) - self.all_stopwords

                if not words1 or not words2:
                    return 0.0

                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))

                return intersection / union if union > 0 else 0.0
            except Exception as e2:
                print(f"Ошибка в fallback методе: {e2}")
                return 0.0


class LLMTestCaseEvaluator:
    """Главный класс для оценки качества тест-кейсов от LLM"""

    def __init__(self):
        self.parser = TestCaseParser()
        self.analyzer = QualityAnalyzer()

    def evaluate_llm_responses(self, responses_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Оценка ответов LLM"""
        results = []

        for i, response in enumerate(responses_data):
            try:
                llm_name = response.get('llm_name', 'Unknown')
                function_type = response.get('function_type', None)
                response_text = response.get('response_text', '')

                print(f"Обрабатываем ответ {i + 1}/{len(responses_data)}: {llm_name}")

                test_cases = self.parser.parse_test_cases(response_text)
                print(f"  Найдено тест-кейсов: {len(test_cases)}")

                metrics = self.analyzer.analyze_test_cases(test_cases, function_type)

                results.append({
                    'LLM': llm_name,
                    'Function_Type': function_type,
                    'Test_Cases_Count': len(test_cases),
                    'Completeness': metrics.completeness,
                    'Semantic_Relevance': metrics.semantic_relevance,
                    'Diversity': metrics.diversity,
                    'Language_Quality': metrics.language_quality,
                    'Redundancy': metrics.redundancy,
                    'Overall_Score': metrics.overall_score
                })

            except Exception as e:
                print(f"Ошибка при обработке ответа {i + 1}: {e}")
                results.append({
                    'LLM': response.get('llm_name', 'Unknown'),
                    'Function_Type': response.get('function_type', None),
                    'Test_Cases_Count': 0,
                    'Completeness': 0.0,
                    'Semantic_Relevance': 0.0,
                    'Diversity': 0.0,
                    'Language_Quality': 0.0,
                    'Redundancy': 0.0,
                    'Overall_Score': 0.0
                })

        return pd.DataFrame(results)

    def generate_report(self, results_df: pd.DataFrame, output_file: str = None):
        """Генерация отчета с визуализацией"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Анализ качества тест-кейсов LLM', fontsize=16)

        llm_scores = results_df.groupby('LLM')['Overall_Score'].mean().sort_values(ascending=False)
        axes[0, 0].bar(llm_scores.index, llm_scores.values)
        axes[0, 0].set_title('Общий балл качества по LLM')
        axes[0, 0].set_ylabel('Балл')
        axes[0, 0].tick_params(axis='x', rotation=45)

        test_counts = results_df.groupby('LLM')['Test_Cases_Count'].mean()
        axes[0, 1].bar(test_counts.index, test_counts.values, color='orange')
        axes[0, 1].set_title('Среднее количество тест-кейсов')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].tick_params(axis='x', rotation=45)

        metrics_cols = ['Completeness', 'Semantic_Relevance', 'Diversity', 'Language_Quality']
        avg_metrics = results_df[metrics_cols].mean()

        angles = np.linspace(0, 2 * np.pi, len(metrics_cols), endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг

        avg_metrics_list = avg_metrics.tolist()
        avg_metrics_list += avg_metrics_list[:1]

        axes[0, 2] = plt.subplot(2, 3, 3, projection='polar')
        axes[0, 2].plot(angles, avg_metrics_list, 'o-', linewidth=2)
        axes[0, 2].fill(angles, avg_metrics_list, alpha=0.25)
        axes[0, 2].set_xticks(angles[:-1])
        axes[0, 2].set_xticklabels(['Полнота', 'Релевантность', 'Разнообразие', 'Качество языка'])
        axes[0, 2].set_title('Средние метрики качества')

        correlation_matrix = results_df[metrics_cols + ['Overall_Score']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Корреляция метрик')

        axes[1, 1].hist(results_df['Redundancy'], bins=10, alpha=0.7, color='red')
        axes[1, 1].set_title('Распределение избыточности')
        axes[1, 1].set_xlabel('Уровень избыточности')
        axes[1, 1].set_ylabel('Частота')

        if 'Function_Type' in results_df.columns and results_df['Function_Type'].notna().any():
            function_scores = results_df.groupby('Function_Type')['Overall_Score'].mean()
            axes[1, 2].bar(function_scores.index, function_scores.values, color='green')
            axes[1, 2].set_title('Качество по типам функций')
            axes[1, 2].set_ylabel('Балл')
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, 'Данные о типах\nфункций отсутствуют',
                            ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Типы функций')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        print("=" * 80)
        print("ОТЧЕТ ПО АНАЛИЗУ КАЧЕСТВА ТЕСТ-КЕЙСОВ LLM")
        print("=" * 80)

        print(f"\nОбщая статистика:")
        print(f"Всего проанализировано ответов: {len(results_df)}")
        print(f"Уникальных LLM: {results_df['LLM'].nunique()}")
        print(f"Среднее количество тест-кейсов: {results_df['Test_Cases_Count'].mean():.1f}")

        print(f"\nРейтинг LLM по общему качеству:")
        for i, (llm, score) in enumerate(llm_scores.items(), 1):
            print(f"{i}. {llm}: {score:.3f}")

        print(f"\nСредние значения метрик:")
        for metric in metrics_cols:
            avg_val = results_df[metric].mean()
            print(f"- {metric}: {avg_val:.3f}")
        print(f"- Redundancy: {results_df['Redundancy'].mean():.3f}")
        print(f"- Overall Score: {results_df['Overall_Score'].mean():.3f}")

        print(f"\nДетальный анализ по LLM:")
        for llm in results_df['LLM'].unique():
            llm_data = results_df[results_df['LLM'] == llm]
            print(f"\n{llm}:")
            print(f"  Количество ответов: {len(llm_data)}")
            print(f"  Среднее количество тест-кейсов: {llm_data['Test_Cases_Count'].mean():.1f}")
            print(f"  Полнота: {llm_data['Completeness'].mean():.3f}")
            print(f"  Релевантность: {llm_data['Semantic_Relevance'].mean():.3f}")
            print(f"  Разнообразие: {llm_data['Diversity'].mean():.3f}")
            print(f"  Качество языка: {llm_data['Language_Quality'].mean():.3f}")
            print(f"  Избыточность: {llm_data['Redundancy'].mean():.3f}")
            print(f"  Общий балл: {llm_data['Overall_Score'].mean():.3f}")

    def save_detailed_results(self, results_df: pd.DataFrame, filename: str = "llm_analysis_results.xlsx"):
        """Сохранение детальных результатов в Excel"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Results', index=False)

            llm_summary = results_df.groupby('LLM').agg({
                'Test_Cases_Count': ['count', 'mean', 'std'],
                'Completeness': 'mean',
                'Semantic_Relevance': 'mean',
                'Diversity': 'mean',
                'Language_Quality': 'mean',
                'Redundancy': 'mean',
                'Overall_Score': 'mean'
            }).round(3)
            llm_summary.to_excel(writer, sheet_name='LLM_Summary')

            metrics_cols = ['Completeness', 'Semantic_Relevance', 'Diversity', 'Language_Quality', 'Redundancy',
                            'Overall_Score']
            correlation_matrix = results_df[metrics_cols].corr()
            correlation_matrix.to_excel(writer, sheet_name='Correlations')

        print(f"Детальные результаты сохранены в файл: {filename}")


def load_data_from_json(json_file: str) -> List[Dict[str, Any]]:
    """Загрузка данных из JSON файла"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            converted_data = []
            for llm_name, llm_data in data.items():
                if isinstance(llm_data, dict):
                    for function_type, response_text in llm_data.items():
                        converted_data.append({
                            'llm_name': llm_name,
                            'function_type': function_type,
                            'response_text': response_text
                        })
                else:
                    converted_data.append({
                        'llm_name': llm_name,
                        'function_type': None,
                        'response_text': llm_data
                    })
            return converted_data

        return data

    except FileNotFoundError:
        print(f"Файл {json_file} не найден")
        return []
    except json.JSONDecodeError as e:
        print(f"Ошибка при парсинге JSON: {e}")
        return []


def main():
    print("Анализатор качества тест-кейсов LLM")
    print("=" * 50)

    evaluator = LLMTestCaseEvaluator()

    data = load_data_from_json("data.json")

    if not data:
        print("Нет данных для анализа")
        return

    print(f"Загружено {len(data)} ответов LLM для анализа")

    results_df = evaluator.evaluate_llm_responses(data)

    evaluator.generate_report(results_df, "llm_analysis_report.png")

    evaluator.save_detailed_results(results_df, "llm_testcases_analysis.xlsx")

    print("\nАнализ завершен!")
    print("Результаты сохранены в:")
    print("- llm_analysis_report.png (визуализация)")
    print("- llm_testcases_analysis.xlsx (детальные данные)")


if __name__ == "__main__":
    main()