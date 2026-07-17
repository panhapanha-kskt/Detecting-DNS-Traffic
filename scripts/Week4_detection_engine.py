#!/usr/bin/env python3
"""
week4_detection_engine.py — Unified DNS Detection Engine (Updated for Week 5)
Combines:
  - Microsoft Sentinel DNS Rules (4 rules, unchanged)
  - Z-score statistical detection (unchanged)
  - NEW: ML scores from ml_bridge.py (ml_score_A, ml_score_B, ml_combined_score)
  - Microsoft Defender Threat Intelligence API (optional, unchanged)

CHANGES from original week4:
  - Accepts week3_features_with_ml.json (output of ml_bridge.py)
  - Adds Rule 5: ML-based detection (uses ml_combined_label + ml_confidence)
  - Adds Rule 6: ML threat score threshold (uses ml_threat_score)
  - All original Week 3 + Week 4 features/columns are UNTOUCHED
  - New alert fields: ml_score_A, ml_score_B, ml_combined_score, ml_confidence

Usage:
  # With ML scores (recommended — run ml_bridge.py first)
  python week4_detection_engine.py \\
    --input data/week3_features_with_ml.json \\
    --output data/week4_unified_alerts.json

  # Without ML scores (original mode — backward compatible)
  python week4_detection_engine.py \\
    --input data/week3_features_all.json \\
    --output data/week4_unified_alerts.json
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import uuid
import math
from collections import defaultdict
import argparse
import requests
from tqdm import tqdm
import pytz
from dateutil.parser import parse


# ============================================
# MICROSOFT DEFENDER THREAT INTELLIGENCE API
# (UNCHANGED from original Week 4)
# ============================================

class MDTIIntegration:
    """Microsoft Defender Threat Intelligence API Integration"""

    def __init__(self, client_id=None, client_secret=None, tenant_id=None):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.tenant_id     = tenant_id
        self.access_token  = None
        self.token_expiry  = None

    def set_credentials(self, client_id, client_secret, tenant_id):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.tenant_id     = tenant_id

    def get_access_token(self):
        if not all([self.client_id, self.client_secret, self.tenant_id]):
            print("⚠ MDTI credentials not configured. Skipping API calls.")
            return None
        if self.access_token and self.token_expiry:
            if datetime.now() < self.token_expiry - timedelta(minutes=5):
                return self.access_token

        token_url  = f'https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token'
        token_data = {
            'grant_type':    'client_credentials',
            'client_id':     self.client_id,
            'client_secret': self.client_secret,
            'scope':         'https://graph.microsoft.com/.default'
        }
        try:
            token_r = requests.post(token_url, data=token_data, timeout=10)
            if token_r.status_code == 200:
                token_json         = token_r.json()
                self.access_token  = token_json.get("access_token")
                expires_in         = token_json.get("expires_in", 3600)
                self.token_expiry  = datetime.now() + timedelta(seconds=expires_in)
                return self.access_token
            else:
                print(f"⚠ Failed to get access token: {token_r.status_code}")
                return None
        except Exception as e:
            print(f"⚠ Error getting access token: {e}")
            return None

    def resolve_dns(self, ip):
        access_token = self.get_access_token()
        if not access_token:
            return False, "API authentication failed"
        url     = f'https://graph.microsoft.com/v1.0/security/threatIntelligence/hosts/{ip}/passiveDns'
        headers = {'Authorization': 'Bearer ' + access_token}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data    = response.json()
                records = data.get('value', [])
                if not records:
                    return False, "No DNS records found"
                for record in records:
                    artifact = record.get('artifact', {})
                    hostname = artifact.get('id', '')
                    if any(bad in hostname for bad in ['zpath.net', 'malware', 'bad']):
                        return False, f"Contains suspicious pattern: {hostname}"
                most_recent = max(records, key=lambda r: r.get('lastSeenDateTime', ''))
                hostname    = most_recent.get('artifact', {}).get('id', 'Unknown hostname')
                return True, hostname
            else:
                return False, f"API returned status code {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "Request timeout"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def get_host_reputation(self, ip):
        access_token = self.get_access_token()
        if not access_token:
            return "Unknown (API auth failed)"
        url     = f'https://graph.microsoft.com/v1.0/security/threatIntelligence/hosts/{ip}/reputation'
        headers = {'Authorization': 'Bearer ' + access_token}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data            = response.json()
                reputation_info = data.get('value', {})
                if isinstance(reputation_info, list) and len(reputation_info) > 0:
                    reputation_info = reputation_info[0]
                reputation     = reputation_info.get('reputation', 'Unknown')
                classification = reputation_info.get('classification', '')
                return f"{reputation} ({classification})" if classification else reputation
            else:
                return f"API error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    def enrich_ip_data(self, ip_list, show_progress=True):
        if not ip_list:
            return {}, {}
        print(f"\n🔍 Enriching {len(ip_list)} IPs with Microsoft Defender TI...")
        dns_results        = {}
        reputation_results = {}
        iterator = tqdm(ip_list, desc="Enriching IPs", unit="IP") if show_progress else ip_list
        for ip in iterator:
            success, hostname = self.resolve_dns(ip)
            if success:
                dns_results[ip] = hostname
            reputation_results[ip] = self.get_host_reputation(ip)
        return dns_results, reputation_results


# ============================================
# Z-SCORE STATISTICAL DETECTION (UNCHANGED)
# ============================================

class ZScoreDetector:
    """Z-score statistical detection from NetSleuthXplorer"""

    def __init__(self, zscore_threshold=2.0):
        self.zscore_threshold = zscore_threshold
        self.baseline_stats   = {}

    def calculate_zscore(self, value, mean, std):
        if std == 0 or pd.isna(std) or std is None:
            return 0
        return (value - mean) / std

    def calculate_zscore_df(self, dataframe, value_column, group_columns):
        df_grouped = dataframe.groupby(group_columns)[value_column].agg(['mean', 'std']).reset_index()
        df_merged  = dataframe.merge(df_grouped, on=group_columns, how='left')
        df_merged['z_score'] = df_merged.apply(
            lambda row: ((row[value_column] - row['mean']) / row['std'])
                        if row['std'] != 0 else np.nan, axis=1)
        return df_merged

    def build_baseline(self, df, features, group_by='client_ip'):
        print("\n📊 Building statistical baseline for Z-score detection...")
        baseline = {}
        for ip in df[group_by].unique():
            ip_df          = df[df[group_by] == ip]
            baseline[ip]   = {}
            for feature in features:
                if feature in ip_df.columns:
                    baseline[ip][feature] = {
                        'mean':  float(ip_df[feature].mean()) if not pd.isna(ip_df[feature].mean()) else 0,
                        'std':   float(ip_df[feature].std())  if not pd.isna(ip_df[feature].std())  else 1,
                        'count': len(ip_df),
                        'min':   float(ip_df[feature].min())  if not pd.isna(ip_df[feature].min())  else 0,
                        'max':   float(ip_df[feature].max())  if not pd.isna(ip_df[feature].max())  else 0,
                    }
        print(f"✅ Built baseline for {len(baseline)} unique IPs")
        self.baseline_stats = baseline
        return baseline

    def detect_anomalies(self, df, features, group_by='client_ip'):
        if not self.baseline_stats:
            print("⚠ No baseline statistics available. Run build_baseline first.")
            return df
        result_df = df.copy()
        for feature in features:
            if feature not in result_df.columns:
                continue
            result_df[f'{feature}_zscore'] = result_df.apply(
                lambda row: self.calculate_zscore(
                    row[feature],
                    self.baseline_stats.get(row.get(group_by), {}).get(feature, {}).get('mean', 0),
                    self.baseline_stats.get(row.get(group_by), {}).get(feature, {}).get('std', 1)
                ) if row.get(group_by) in self.baseline_stats else 0,
                axis=1
            )
            result_df[f'{feature}_anomaly'] = np.abs(result_df[f'{feature}_zscore']) > self.zscore_threshold
        return result_df

    def get_anomaly_alerts(self, df_with_zscores, features, source_file=None):
        alerts = []
        for feature in features:
            anomaly_col = f'{feature}_anomaly'
            zscore_col  = f'{feature}_zscore'
            if anomaly_col not in df_with_zscores.columns:
                continue
            anomaly_df = df_with_zscores[df_with_zscores[anomaly_col] == True].copy()
            for _, row in anomaly_df.iterrows():
                ip       = row.get('client_ip', 'unknown')
                baseline = self.baseline_stats.get(ip, {}).get(feature, {})
                alert    = {
                    'alert_id':      str(uuid.uuid4()),
                    'rule_name':     f'Z-score Statistical Detection - {feature}',
                    'rule_type':     'statistical',
                    'rule_source':   'NetSleuthXplorer + Microsoft Sentinel',
                    'timestamp':     datetime.now().isoformat(),
                    'query_time':    row.get('timestamp', ''),
                    'client_ip':     ip,
                    'domain':        row.get('domain', 'unknown'),
                    'feature':       feature,
                    'value':         float(row[feature]) if feature in row else 0,
                    'zscore':        float(row[zscore_col]) if zscore_col in row else 0,
                    'threshold':     self.zscore_threshold,
                    'baseline_mean': baseline.get('mean', 0),
                    'baseline_std':  baseline.get('std', 1),
                    'severity':      'High' if abs(row[zscore_col]) > 3 else 'Medium',
                    'mitre_tactics':    ['Discovery', 'CommandAndControl'],
                    'mitre_techniques': ['T1046', 'T1568'],
                    'description':   (f"Statistical anomaly: {feature} Z-score = "
                                      f"{row[zscore_col]:.2f} (value: {row[feature]}, "
                                      f"baseline mean: {baseline.get('mean', 0):.2f})"),
                    'source_file':   source_file,
                }
                alerts.append(alert)
        return alerts


# ============================================
# RULE-BASED DETECTION (UPDATED)
# ============================================

class RuleBasedDNSDetector:
    """
    Week 4 (updated): Rule-based + Statistical + ML detection.
    Original 4 rules + Z-score UNCHANGED.
    Added: Rule 5 (ML combined label) + Rule 6 (ML threat score).
    """

    def __init__(self, config_file=None, mdt_integration=None, zscore_detector=None):
        # ── ORIGINAL RULES (UNCHANGED) ────────────────────────────────
        self.nxdomain_rule = {
            'name':        'Excessive NXDOMAIN DNS Queries',
            'description': 'Detect excessive DNS queries to non-existent domains (C2 communication)',
            'threshold':   200,
            'time_window': '15min',
            'lookback':    '1h',
            'severity':    'Medium',
            'tactics':     ['CommandAndControl'],
            'techniques':  ['T1568', 'T1008'],
            'source':      'Microsoft Sentinel ASIM DNS Solution'
        }
        self.multi_client_rule = {
            'name':        'Multiple Errors for Same DNS Query',
            'description': 'Detect multiple clients reporting errors for same domain (malware beaconing)',
            'threshold':   2,
            'time_window': '10min',
            'error_codes': ['NXDOMAIN', 'SERVFAIL', 'REFUSED'],
            'severity':    'Medium',
            'tactics':     ['CommandAndControl'],
            'techniques':  ['T1568', 'T1573', 'T1008'],
            'source':      'Microsoft Sentinel ASIM DNS Solution'
        }
        self.tunneling_rule = {
            'name':             'DNS Tunneling Detection',
            'description':      'Detect DNS tunneling via long queries and high entropy',
            'length_threshold': 100,
            'entropy_threshold': 4.0,
            'severity':         'High',
            'tactics':          ['Exfiltration'],
            'techniques':       ['T1572', 'T1048'],
            'source':           'Project Requirement'
        }
        self.subdomain_rule = {
            'name':            'Deep Subdomain Detection',
            'description':     'Detect deep subdomain structures (tunneling/malware)',
            'depth_threshold': 5,
            'severity':        'Medium',
            'tactics':         ['CommandAndControl'],
            'techniques':      ['T1572'],
            'source':          'Project Requirement'
        }

        # ── NEW ML RULES ──────────────────────────────────────────────
        self.ml_label_rule = {
            'name':        'ML Model Attack Detection',
            'description': 'ML ensemble (RF + Transformer + Trafficformer) flagged this query',
            'severity':    'High',
            'tactics':     ['CommandAndControl', 'Exfiltration'],
            'techniques':  ['T1572', 'T1568', 'T1048'],
            'source':      'ML Bridge (Path A + Path B)'
        }
        self.ml_score_rule = {
            'name':           'ML High Threat Score',
            'description':    'Query has high combined ML threat score',
            'score_threshold': 0.7,   # flag if ml_threat_score >= 0.7
            'severity':       'High',
            'tactics':        ['CommandAndControl', 'Exfiltration'],
            'techniques':     ['T1572', 'T1568'],
            'source':         'ML Bridge (combined threat score)'
        }

        self.stats = {
            'total_queries_analyzed': 0,
            'alerts_by_rule':         defaultdict(int),
            'alerts_by_severity':     defaultdict(int),
            'top_offending_ips':      defaultdict(int),
            'top_suspicious_domains': defaultdict(int),
        }

        self.alerts           = []
        self.mdt_integration  = mdt_integration
        self.zscore_detector  = zscore_detector
        self.has_ml_scores    = False   # set to True in run_detection if ML cols present

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

        self.print_init_banner()

    def print_init_banner(self):
        print("\n" + "="*80)
        print("WEEK 4 (UPDATED) — UNIFIED DNS DETECTION ENGINE")
        print("="*80)
        for rule in [self.nxdomain_rule, self.multi_client_rule,
                     self.tunneling_rule, self.subdomain_rule]:
            print(f"Rule: {rule['name']}  [{rule['source']}]")
        print(f"\nML Rules (active if ml_bridge.py output is used):")
        print(f"  Rule 5: {self.ml_label_rule['name']}")
        print(f"  Rule 6: {self.ml_score_rule['name']}  (threshold: {self.ml_score_rule['score_threshold']})")
        if self.zscore_detector:
            print(f"\nZ-score threshold: {self.zscore_detector.zscore_threshold} σ  [NetSleuthXplorer]")
        mdti_on = (self.mdt_integration and
                   all([self.mdt_integration.client_id,
                        self.mdt_integration.client_secret,
                        self.mdt_integration.tenant_id]))
        print(f"Microsoft Defender TI: {'Enabled' if mdti_on else 'Disabled'}")
        print("="*80)

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            if 'nxdomain_threshold'  in config: self.nxdomain_rule['threshold']          = config['nxdomain_threshold']
            if 'multi_client_threshold' in config: self.multi_client_rule['threshold']   = config['multi_client_threshold']
            if 'tunneling_length'    in config: self.tunneling_rule['length_threshold']  = config['tunneling_length']
            if 'tunneling_entropy'   in config: self.tunneling_rule['entropy_threshold'] = config['tunneling_entropy']
            if 'subdomain_depth'     in config: self.subdomain_rule['depth_threshold']   = config['subdomain_depth']
            if 'ml_score_threshold'  in config: self.ml_score_rule['score_threshold']    = config['ml_score_threshold']
            if 'zscore_threshold'    in config and self.zscore_detector:
                self.zscore_detector.zscore_threshold = config['zscore_threshold']
            print(f"\n✅ Loaded configuration from: {config_file}")
        except Exception as e:
            print(f"⚠ Error loading config: {e}")

    def load_week3_features(self, features_file):
        """
        Load Week 3 or ml_bridge output features.
        Auto-detects whether ML columns are present.
        """
        print(f"\n📂 Loading features from: {features_file}")
        try:
            if features_file.endswith('.json'):
                with open(features_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                df = pd.read_csv(features_file)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Detect if ML bridge output
            ml_cols = ['ml_score_A', 'ml_score_B', 'ml_combined_score']
            self.has_ml_scores = any(c in df.columns for c in ml_cols)
            if self.has_ml_scores:
                print(f"✅ ML scores detected — ML rules (5 & 6) will be active")
            else:
                print(f"ℹ No ML scores found — run ml_bridge.py first to enable ML rules")

            print(f"✅ Loaded {len(df)} DNS queries")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            return None

    def calculate_entropy(self, domain):
        if not domain or pd.isna(domain):
            return 0.0
        domain = str(domain)
        if len(domain) == 0:
            return 0.0
        char_freq = {}
        for char in domain:
            char_freq[char] = char_freq.get(char, 0) + 1
        entropy = 0.0
        for count in char_freq.values():
            p = count / len(domain)
            entropy -= p * math.log2(p)
        return round(entropy, 4)

    def enrich_alerts_with_threat_intel(self, alerts):
        if not self.mdt_integration:
            return alerts
        unique_ips = set()
        for alert in alerts:
            if alert.get('client_ip') and alert['client_ip'] not in ['multiple', 'unknown']:
                unique_ips.add(alert['client_ip'])
        if not unique_ips:
            return alerts
        dns_results, reputation_results = self.mdt_integration.enrich_ip_data(list(unique_ips))
        for alert in alerts:
            ip = alert.get('client_ip')
            if ip in dns_results:
                alert['enriched_dns'] = dns_results[ip]
            if ip in reputation_results:
                alert['enriched_reputation'] = reputation_results[ip]
                rep = reputation_results[ip].lower()
                if any(bad in rep for bad in ['malicious', 'bad', 'suspicious']):
                    alert['threat_score'] = alert.get('threat_score', 0) + 50
                    alert['severity']     = 'High'
        return alerts

    # ── ORIGINAL RULE 1 (UNCHANGED) ───────────────────────────────────
    def rule1_excessive_nxdomain(self, df):
        print(f"\n🔍 [Rule 1] Running: {self.nxdomain_rule['name']}")
        if 'is_nxdomain' not in df.columns:
            if 'response_code' in df.columns:
                df['is_nxdomain'] = df['response_code'].str.upper().isin(['NXDOMAIN', '3'])
            else:
                print("   ⚠ Cannot detect NXDOMAIN - missing required columns")
                return []

        nxdomain_df = df[df['is_nxdomain'] == True].copy()
        print(f"   📊 Found {len(nxdomain_df)} NXDOMAIN responses")
        if len(nxdomain_df) == 0:
            return []

        nxdomain_df['time_window'] = nxdomain_df['timestamp'].dt.floor('15min')
        grouped = nxdomain_df.groupby(['client_ip', 'time_window']).agg(
            nxdomain_count=('domain', 'count'),
            sample_domains=('domain', lambda x: list(x)[:20])
        ).reset_index()

        threshold = self.nxdomain_rule['threshold']
        alerts_df = grouped[grouped['nxdomain_count'] > threshold]
        print(f"   🚨 Alerts generated: {len(alerts_df)}")

        alerts = []
        for _, row in alerts_df.iterrows():
            severity = 'High' if row['nxdomain_count'] > threshold * 2 else 'Medium'
            alert = self.create_alert(
                rule_name   = self.nxdomain_rule['name'],
                rule_source = self.nxdomain_rule['source'],
                severity    = severity,
                client_ip   = row['client_ip'],
                domain      = 'N/A',
                details     = {
                    'nxdomain_count': int(row['nxdomain_count']),
                    'threshold':      threshold,
                    'time_window':    row['time_window'].isoformat(),
                    'sample_domains': row['sample_domains'][:10],
                },
                description = (f"Client {row['client_ip']} generated {row['nxdomain_count']} "
                               f"NXDOMAIN responses in 15 minutes (threshold: {threshold})"),
                tactics     = self.nxdomain_rule['tactics'],
                techniques  = self.nxdomain_rule['techniques'],
            )
            alerts.append(alert)
        return alerts

    # ── ORIGINAL RULE 2 (UNCHANGED) ───────────────────────────────────
    def rule2_multiple_clients_same_error(self, df):
        print(f"\n🔍 [Rule 2] Running: {self.multi_client_rule['name']}")
        error_codes = self.multi_client_rule['error_codes']
        if 'response_code' not in df.columns:
            print("   ⚠ Cannot detect error responses - missing 'response_code' column")
            return []

        error_df = df[df['response_code'].str.upper().isin(error_codes)].copy()
        print(f"   📊 Found {len(error_df)} error responses")
        if len(error_df) == 0:
            return []

        error_df['time_window'] = error_df['timestamp'].dt.floor('10min')
        grouped = error_df.groupby(['domain', 'time_window']).agg(
            unique_clients=('client_ip', 'nunique')
        ).reset_index()

        threshold = self.multi_client_rule['threshold']
        alerts_df = grouped[grouped['unique_clients'] >= threshold]
        print(f"   🚨 Alerts generated: {len(alerts_df)}")

        alerts = []
        for _, row in alerts_df.iterrows():
            sample_ips = error_df[
                (error_df['domain'] == row['domain']) &
                (error_df['time_window'] == row['time_window'])
            ]['client_ip'].unique()[:10]

            if   row['unique_clients'] >= 10: severity = 'High'
            elif row['unique_clients'] >= 5:  severity = 'Medium'
            else:                              severity = 'Low'

            alert = self.create_alert(
                rule_name   = self.multi_client_rule['name'],
                rule_source = self.multi_client_rule['source'],
                severity    = severity,
                client_ip   = 'multiple',
                domain      = row['domain'],
                details     = {
                    'unique_clients': int(row['unique_clients']),
                    'threshold':      threshold,
                    'time_window':    row['time_window'].isoformat(),
                    'sample_clients': list(sample_ips),
                },
                description = (f"Domain '{row['domain']}' queried with errors by "
                               f"{row['unique_clients']} different clients"),
                tactics     = self.multi_client_rule['tactics'],
                techniques  = self.multi_client_rule['techniques'],
            )
            alerts.append(alert)
        return alerts

    # ── ORIGINAL RULE 3 (UNCHANGED) ───────────────────────────────────
    def rule3_dns_tunneling_detection(self, df):
        print(f"\n🔍 [Rule 3] Running: {self.tunneling_rule['name']}")
        if 'query_length' not in df.columns and 'domain' in df.columns:
            df['query_length'] = df['domain'].str.len()
        if 'entropy' not in df.columns and 'domain' in df.columns:
            df['entropy'] = df['domain'].apply(self.calculate_entropy)

        length_threshold  = self.tunneling_rule['length_threshold']
        entropy_threshold = self.tunneling_rule['entropy_threshold']
        tunneling_df      = df[
            (df.get('query_length', pd.Series(0, index=df.index)) > length_threshold) |
            (df.get('entropy',       pd.Series(0, index=df.index)) > entropy_threshold)
        ].copy()
        print(f"   📊 Found {len(tunneling_df)} potential tunneling queries")
        if len(tunneling_df) == 0:
            return []

        alerts = []
        for _, row in tunneling_df.iterrows():
            reasons = []
            if row.get('query_length', 0) > length_threshold:
                reasons.append(f"length={row['query_length']}>{length_threshold}")
            if row.get('entropy', 0) > entropy_threshold:
                reasons.append(f"entropy={row['entropy']:.2f}>{entropy_threshold}")

            alert = self.create_alert(
                rule_name   = self.tunneling_rule['name'],
                rule_source = self.tunneling_rule['source'],
                severity    = 'High',
                client_ip   = row.get('client_ip', 'unknown'),
                domain      = row.get('domain', 'unknown'),
                details     = {
                    'query_length':      int(row.get('query_length', 0)),
                    'entropy':           float(row.get('entropy', 0)),
                    'length_threshold':  length_threshold,
                    'entropy_threshold': entropy_threshold,
                    'reasons':           reasons,
                    # Include ML context if available
                    'ml_score_A':     float(row.get('ml_score_A', 0)) if self.has_ml_scores else None,
                    'ml_score_B':     float(row.get('ml_score_B', 0)) if self.has_ml_scores else None,
                    'ml_confidence':  str(row.get('ml_confidence', '')) if self.has_ml_scores else None,
                },
                description = f"DNS tunneling detected: {row.get('domain', 'unknown')} ({', '.join(reasons)})",
                tactics     = self.tunneling_rule['tactics'],
                techniques  = self.tunneling_rule['techniques'],
            )
            alerts.append(alert)
            if len(alerts) >= 100:
                break

        print(f"   🚨 Alerts generated: {len(alerts)}")
        return alerts

    # ── ORIGINAL RULE 4 (UNCHANGED) ───────────────────────────────────
    def rule4_deep_subdomain_detection(self, df):
        print(f"\n🔍 [Rule 4] Running: {self.subdomain_rule['name']}")
        if 'subdomain_depth' not in df.columns and 'domain' in df.columns:
            df['subdomain_depth'] = df['domain'].str.count(r'\.')
        if 'subdomain_depth' not in df.columns:
            print("   ⚠ Cannot detect deep subdomains - missing column")
            return []

        depth_threshold = self.subdomain_rule['depth_threshold']
        deep_df         = df[df['subdomain_depth'] > depth_threshold].copy()
        print(f"   📊 Found {len(deep_df)} queries with deep subdomains")
        if len(deep_df) == 0:
            return []

        grouped = deep_df.groupby('domain').agg(
            sample_clients=('client_ip', lambda x: list(set(x))[:5]),
            subdomain_depth=('subdomain_depth', 'first'),
            timestamp=('timestamp', 'first'),
        ).reset_index()

        alerts = []
        for _, row in grouped.iterrows():
            alert = self.create_alert(
                rule_name   = self.subdomain_rule['name'],
                rule_source = self.subdomain_rule['source'],
                severity    = 'Medium',
                client_ip   = 'multiple',
                domain      = row['domain'],
                details     = {
                    'subdomain_depth': int(row['subdomain_depth']),
                    'threshold':       depth_threshold,
                    'sample_clients':  row['sample_clients'][:5],
                },
                description = (f"Deep subdomain structure: {row['domain']} "
                               f"({row['subdomain_depth']} levels)"),
                tactics     = self.subdomain_rule['tactics'],
                techniques  = self.subdomain_rule['techniques'],
            )
            alerts.append(alert)
            if len(alerts) >= 50:
                break

        print(f"   🚨 Alerts generated: {len(alerts)}")
        return alerts

    # ── NEW RULE 5: ML COMBINED LABEL ─────────────────────────────────
    def rule5_ml_attack_detection(self, df):
        """
        Rule 5 (NEW): Flag queries where ML ensemble fired.
        Uses ml_combined_label from ml_bridge.py output.
        Only active if ML columns are present.
        """
        print(f"\n🔍 [Rule 5] Running: {self.ml_label_rule['name']}")
        if not self.has_ml_scores:
            print("   ⚠ Skipped — no ML scores in data (run ml_bridge.py first)")
            return []

        if 'ml_combined_label' not in df.columns:
            print("   ⚠ Skipped — 'ml_combined_label' column not found")
            return []

        ml_df = df[df['ml_combined_label'] == 1].copy()
        print(f"   📊 Found {len(ml_df)} ML-flagged queries")
        if len(ml_df) == 0:
            return []

        alerts = []
        for _, row in ml_df.iterrows():
            confidence = str(row.get('ml_confidence', 'UNKNOWN'))
            severity   = 'High' if confidence == 'HIGH' else 'Medium'

            alert = self.create_alert(
                rule_name   = self.ml_label_rule['name'],
                rule_source = self.ml_label_rule['source'],
                severity    = severity,
                client_ip   = row.get('client_ip', 'unknown'),
                domain      = row.get('domain', 'unknown'),
                details     = {
                    'ml_score_A':          float(row.get('ml_score_A', 0)),
                    'ml_score_B':          float(row.get('ml_score_B', 0)),
                    'ml_combined_score':   float(row.get('ml_combined_score', 0)),
                    'ml_confidence':       confidence,
                    'ml_verdict_A':        str(row.get('ml_verdict_A', '')),
                    'ml_combined_verdict': str(row.get('ml_combined_verdict', '')),
                    # Week 3 attack context
                    'attack_signature':    str(row.get('attack_signature', 'NORMAL')),
                    'attack_count':        int(row.get('attack_count', 0)),
                },
                description = (f"ML ensemble flagged: {row.get('domain', 'unknown')} "
                               f"(score={row.get('ml_combined_score', 0):.3f}, "
                               f"confidence={confidence})"),
                tactics     = self.ml_label_rule['tactics'],
                techniques  = self.ml_label_rule['techniques'],
            )
            alerts.append(alert)
            if len(alerts) >= 200:
                break

        print(f"   🚨 Alerts generated: {len(alerts)}")
        return alerts

    # ── NEW RULE 6: ML THREAT SCORE ───────────────────────────────────
    def rule6_ml_threat_score(self, df):
        """
        Rule 6 (NEW): Flag queries with high combined ML threat score.
        ml_threat_score = 0.4*rule_score + 0.35*ml_score_A + 0.25*ml_score_B
        Catches cases where BOTH rule-based AND ML models agree.
        """
        print(f"\n🔍 [Rule 6] Running: {self.ml_score_rule['name']}")
        if not self.has_ml_scores:
            print("   ⚠ Skipped — no ML scores in data")
            return []

        if 'ml_threat_score' not in df.columns:
            print("   ⚠ Skipped — 'ml_threat_score' column not found")
            return []

        threshold  = self.ml_score_rule['score_threshold']
        high_df    = df[df['ml_threat_score'] >= threshold].copy()
        print(f"   📊 Found {len(high_df)} queries with threat score >= {threshold}")
        if len(high_df) == 0:
            return []

        alerts = []
        for _, row in high_df.iterrows():
            score    = float(row['ml_threat_score'])
            severity = 'High' if score >= 0.85 else 'Medium'

            alert = self.create_alert(
                rule_name   = self.ml_score_rule['name'],
                rule_source = self.ml_score_rule['source'],
                severity    = severity,
                client_ip   = row.get('client_ip', 'unknown'),
                domain      = row.get('domain', 'unknown'),
                details     = {
                    'ml_threat_score':   score,
                    'score_threshold':   threshold,
                    'ml_score_A':        float(row.get('ml_score_A', 0)),
                    'ml_score_B':        float(row.get('ml_score_B', 0)),
                    'attack_count':      int(row.get('attack_count', 0)),
                    'attack_signature':  str(row.get('attack_signature', 'NORMAL')),
                },
                description = (f"High ML threat score: {row.get('domain', 'unknown')} "
                               f"(threat_score={score:.3f} >= {threshold})"),
                tactics     = self.ml_score_rule['tactics'],
                techniques  = self.ml_score_rule['techniques'],
            )
            alerts.append(alert)
            if len(alerts) >= 200:
                break

        print(f"   🚨 Alerts generated: {len(alerts)}")
        return alerts

    def create_alert(self, rule_name, rule_source, severity, client_ip,
                     domain, details, description, tactics, techniques):
        alert = {
            'alert_id':              str(uuid.uuid4()),
            'timestamp':             datetime.now().isoformat(),
            'rule_name':             rule_name,
            'rule_source':           rule_source,
            'severity':              severity,
            'client_ip':             client_ip,
            'domain':                domain,
            'details':               details,
            'description':           description,
            'mitre_tactics':         tactics,
            'mitre_techniques':      techniques,
            'investigation_priority': 'High' if severity == 'High' else 'Normal',
        }
        self.alerts.append(alert)
        self.stats['alerts_by_rule'][rule_name]         += 1
        self.stats['alerts_by_severity'][severity]      += 1
        if client_ip not in ['multiple', 'unknown']:
            self.stats['top_offending_ips'][client_ip]  += 1
        self.stats['top_suspicious_domains'][domain]    += 1
        return alert

    def run_detection(self, df, source_file=None):
        print("\n" + "="*80)
        print("DETECTION ENGINE EXECUTION")
        print("="*80)
        self.stats['total_queries_analyzed'] = len(df)
        all_alerts = []

        # Original rules (unchanged)
        all_alerts.extend(self.rule1_excessive_nxdomain(df))
        all_alerts.extend(self.rule2_multiple_clients_same_error(df))
        all_alerts.extend(self.rule3_dns_tunneling_detection(df))
        all_alerts.extend(self.rule4_deep_subdomain_detection(df))

        # Z-score (unchanged)
        if self.zscore_detector and self.zscore_detector.baseline_stats:
            print(f"\n🔍 [Z-score] Running: Statistical Anomaly Detection")
            zscore_features = [f for f in ['entropy', 'query_length', 'subdomain_depth'] if f in df.columns]
            if zscore_features:
                df_z         = self.zscore_detector.detect_anomalies(df, zscore_features)
                zscore_alerts = self.zscore_detector.get_anomaly_alerts(df_z, zscore_features, source_file)
                all_alerts.extend(zscore_alerts)
                print(f"   🚨 Statistical alerts: {len(zscore_alerts)}")

        # NEW: ML rules
        all_alerts.extend(self.rule5_ml_attack_detection(df))
        all_alerts.extend(self.rule6_ml_threat_score(df))

        self.alerts = all_alerts

        # MDTI enrichment (unchanged)
        if self.mdt_integration and all_alerts:
            self.alerts = self.enrich_alerts_with_threat_intel(self.alerts)

        self.print_summary()
        return all_alerts

    def print_summary(self):
        print("\n" + "="*80)
        print("DETECTION SUMMARY")
        print("="*80)
        print(f"Total queries analyzed : {self.stats['total_queries_analyzed']}")
        print(f"Total alerts generated : {len(self.alerts)}")
        ml_enabled = "✅ Active" if self.has_ml_scores else "⚠ Inactive (run ml_bridge.py)"
        print(f"ML Rules (5 & 6)       : {ml_enabled}")
        print("\nAlerts by Rule:")
        for rule, count in self.stats['alerts_by_rule'].items():
            print(f"  - {rule}: {count}")
        print("\nAlerts by Severity:")
        for sev, count in self.stats['alerts_by_severity'].items():
            print(f"  - {sev}: {count}")
        print("\nTop Offending IPs:")
        for ip, count in sorted(self.stats['top_offending_ips'].items(),
                                 key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {ip}: {count}")
        print("\nTop Suspicious Domains:")
        for domain, count in sorted(self.stats['top_suspicious_domains'].items(),
                                     key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {domain}: {count}")
        print("="*80)

    def export_alerts(self, output_file):
        output = {
            'generated':              datetime.now().isoformat(),
            'week':                   4,
            'phase':                  'Unified Detection Engine (Rule + Statistical + ML)',
            'total_queries_analyzed': self.stats['total_queries_analyzed'],
            'total_alerts':           len(self.alerts),
            'ml_rules_active':        self.has_ml_scores,
            'alerts_by_rule':         dict(self.stats['alerts_by_rule']),
            'alerts_by_severity':     dict(self.stats['alerts_by_severity']),
            'rules_used': [
                {'name': self.nxdomain_rule['name'],    'threshold': self.nxdomain_rule['threshold'],    'source': self.nxdomain_rule['source']},
                {'name': self.multi_client_rule['name'],'threshold': self.multi_client_rule['threshold'],'source': self.multi_client_rule['source']},
                {'name': self.tunneling_rule['name'],   'thresholds': {'length': self.tunneling_rule['length_threshold'], 'entropy': self.tunneling_rule['entropy_threshold']}, 'source': self.tunneling_rule['source']},
                {'name': self.subdomain_rule['name'],   'threshold': self.subdomain_rule['depth_threshold'], 'source': self.subdomain_rule['source']},
                {'name': self.ml_label_rule['name'],    'active': self.has_ml_scores, 'source': self.ml_label_rule['source']},
                {'name': self.ml_score_rule['name'],    'threshold': self.ml_score_rule['score_threshold'], 'active': self.has_ml_scores, 'source': self.ml_score_rule['source']},
            ],
            'statistical_detection': {
                'enabled':   self.zscore_detector is not None,
                'threshold': self.zscore_detector.zscore_threshold if self.zscore_detector else None,
                'source':    'NetSleuthXplorer',
            },
            'ml_detection': {
                'enabled': self.has_ml_scores,
                'source':  'ml_bridge.py (Path A: RF+Transformer + Path B: Trafficformer)',
            },
            'threat_intelligence': {
                'enabled': self.mdt_integration is not None and all([
                    self.mdt_integration.client_id,
                    self.mdt_integration.client_secret,
                    self.mdt_integration.tenant_id,
                ]),
                'source': 'Microsoft Defender Threat Intelligence',
            },
            'alerts': self.alerts,
        }
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n✅ Alerts exported to: {output_file}")
        return output_file

    def export_alerts_csv(self, output_file):
        if not self.alerts:
            print("⚠ No alerts to export to CSV")
            return
        flat_alerts = []
        for alert in self.alerts:
            d = alert.get('details', {})
            flat_alerts.append({
                'timestamp':          alert.get('timestamp', ''),
                'rule_name':          alert.get('rule_name', ''),
                'severity':           alert.get('severity', ''),
                'client_ip':          alert.get('client_ip', ''),
                'domain':             alert.get('domain', ''),
                'description':        alert.get('description', ''),
                'mitre_tactics':      ', '.join(alert.get('mitre_tactics', [])),
                'mitre_techniques':   ', '.join(alert.get('mitre_techniques', [])),
                'ml_score_A':         d.get('ml_score_A', ''),
                'ml_score_B':         d.get('ml_score_B', ''),
                'ml_combined_score':  d.get('ml_combined_score', ''),
                'ml_confidence':      d.get('ml_confidence', ''),
                'enriched_reputation': alert.get('enriched_reputation', ''),
                'enriched_dns':       alert.get('enriched_dns', ''),
            })
        pd.DataFrame(flat_alerts).to_csv(output_file, index=False)
        print(f"✅ Alerts CSV exported to: {output_file}")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Week 4 Unified DNS Detection Engine (with ML support)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # With ML scores (recommended)
  python week4_detection_engine.py \\
    --input data/week3_features_with_ml.json \\
    --output data/week4_unified_alerts.json --csv

  # Without ML scores (original mode — backward compatible)
  python week4_detection_engine.py \\
    --input data/week3_features_all.json \\
    --output data/week4_unified_alerts.json
        """
    )
    parser.add_argument('--input',  '-i', required=True,  help='Input features file (JSON or CSV)')
    parser.add_argument('--output', '-o', required=True,  help='Output alerts file')
    parser.add_argument('--config', '-c', default=None,   help='Config file for custom thresholds')
    parser.add_argument('--csv',    action='store_true',  help='Also export alerts as CSV')
    parser.add_argument('--zscore', type=float, default=2.0, help='Z-score threshold (default: 2.0)')
    parser.add_argument('--mdt-client-id',     default=None)
    parser.add_argument('--mdt-client-secret', default=None)
    parser.add_argument('--mdt-tenant-id',     default=None)
    args = parser.parse_args()

    mdt_integration = MDTIIntegration()
    if args.mdt_client_id and args.mdt_client_secret and args.mdt_tenant_id:
        mdt_integration.set_credentials(
            args.mdt_client_id, args.mdt_client_secret, args.mdt_tenant_id
        )

    zscore_detector = ZScoreDetector(zscore_threshold=args.zscore)
    detector        = RuleBasedDNSDetector(
        config_file      = args.config,
        mdt_integration  = mdt_integration,
        zscore_detector  = zscore_detector,
    )

    df = detector.load_week3_features(args.input)
    if df is None:
        print("❌ Failed to load input file. Exiting.")
        return

    zscore_features = [f for f in ['entropy', 'query_length', 'subdomain_depth'] if f in df.columns]
    if zscore_features:
        zscore_detector.build_baseline(df, zscore_features)

    detector.run_detection(df, source_file=args.input)
    detector.export_alerts(args.output)

    if args.csv:
        detector.export_alerts_csv(args.output.replace('.json', '.csv'))

    print("\n✅ Detection complete.")
    print("   Rule 1–4: Microsoft Sentinel rules")
    print("   Z-score:  NetSleuthXplorer statistical detection")
    if detector.has_ml_scores:
        print("   Rule 5–6: ML detection (Path A: RF+Transformer, Path B: Trafficformer)")
    else:
        print("   Rule 5–6: INACTIVE — run ml_bridge.py to enable ML detection")


if __name__ == "__main__":
    main()
